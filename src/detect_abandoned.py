import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

BAG_CLASSES = {"backpack", "handbag", "suitcase"}

def box_of(r):  return (r.x1, r.y1, r.x2, r.y2)
def center(r):  return np.array([(r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2])

# bottom center of a box, for measuring distance to a person
def feet(r):    return np.array([(r.x1 + r.x2) / 2, r.y2])

def iou(a, b):
    """How much two boxes overlap: 0 = not at all, 1 = identical."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)

def snapshot(frame, box):
    """Small grayscale picture of what is inside a box, for comparing later."""
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)

def clothing(frame, r):
    """Color fingerprint of a person's clothes: hue/saturation histogram, head skipped."""
    x1, y1, x2, y2 = [int(v) for v in box_of(r)]
    y1 += (y2 - y1) // 5                       # skip the head
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()

def run(video, mode, dist_limit, time_limit, grace_s, reid):
    df = pd.read_csv(Path("results/tracks") / (Path(video).stem + ".csv"))
    df["is_bag"] = df.cls.isin(BAG_CLASSES)
    by_frame = dict(tuple(df.groupby("frame")))
    empty = df.iloc[0:0]

    # NEW: load the floor map (homography) if this video has been calibrated
    calib = Path("calib") / (Path(video).stem + ".json")
    H = np.array(json.load(open(calib))["H"]) if calib.exists() else None
    unit = "m" if H is not None else "px"
    print(f"distances in {unit}, limit = {dist_limit} {unit}")

    def to_floor(pt):
        """NEW: pixel position -> floor-map position in meters (or pixels if no calibration)."""
        if H is None:
            return pt
        return cv2.perspectiveTransform(np.array([[pt]], np.float32), H)[0, 0]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    GRACE, LIMIT, STILL = int(grace_s * fps), int(time_limit * fps), int(1.0 * fps)

    # Keep track of each bag's history, last seen position of each person, and the state of parked bags
    history = {}
    last_seen, last_pos, person_fp = {}, {}, {}
    parked, alerts = {}, []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = by_frame.get(f, empty)
        people, bags = rows[~rows.is_bag], rows[rows.is_bag]
        for r in people.itertuples():
            last_seen[r.track_id], last_pos[r.track_id] = f, to_floor(feet(r))   # NEW: floor position
            fp = clothing(frame, r)
            if fp is not None:
                person_fp[r.track_id] = fp

        # 1. Check each bag to see if it has been stationary for a while and is not already parked
        for r in bags.itertuples():
            h = history.setdefault(r.track_id, [])
            h.append((f, center(r)))
            recent = np.array([c for (ff, c) in h if ff > f - STILL])
            already = r.track_id in parked or any(iou(b["box"], box_of(r)) > 0.3 for b in parked.values())
            if not already and len(recent) >= 0.6 * STILL and np.ptp(recent, axis=0).max() < 10:
                bag_floor = to_floor(feet(r))                                    # NEW
                near = {p: np.linalg.norm(last_pos[p] - bag_floor)
                        for p in last_seen if f - last_seen[p] <= fps}
                owner = min(near, key=near.get) if near else None
                parked[r.track_id] = dict(box=box_of(r), ref=snapshot(frame, box_of(r)),
                                          owner=owner,
                                          owners={owner} if owner is not None else set(),
                                          owner_fp=person_fp.get(owner),
                                          timer=0, missing=0, state="ATTENDED")
                print(f"[{f/fps:5.1f}s] bag {r.track_id} parked, owner = person {owner}")

        # 2. Check each parked bag to see if it is still there, and if it is attended or abandoned
        for bid, b in list(parked.items()):
            seen = any(iou(b["box"], box_of(r)) > 0.3 for r in bags.itertuples())
            blocked = any(iou(b["box"], box_of(r)) > 0 for r in people.itertuples())
            if seen and not blocked:
                b["ref"] = snapshot(frame, b["box"])          # keep the snapshot fresh
            elif not seen and not blocked:
                seen = np.abs(snapshot(frame, b["box"]) - b["ref"]).mean() < 25
            b["missing"] = 0 if (seen or blocked) else b["missing"] + 1
            if b["missing"] > GRACE:
                print(f"[{f/fps:5.1f}s] bag {bid} gone (picked up)")
                del parked[bid]
                continue

            bag_pt = to_floor(np.array([(b["box"][0] + b["box"][2]) / 2, b["box"][3]]))   # NEW
            if mode == "owner":
                # Spot the owner if the tracker gave them a new ID
                if b["owner_fp"] is not None:
                    for r in people.itertuples():
                        if (r.track_id not in b["owners"] and r.track_id in person_fp and
                                cv2.compareHist(b["owner_fp"], person_fp[r.track_id],
                                                cv2.HISTCMP_CORREL) > reid):
                            b["owners"].add(r.track_id)
                            print(f"[{f/fps:5.1f}s] person {r.track_id} re-identified as owner of bag {bid}")
                # Distance to the closest owner ID seen in the last few seconds
                d = min((np.linalg.norm(last_pos[o] - bag_pt) for o in b["owners"]
                         if f - last_seen[o] <= GRACE), default=np.inf)
            else:   # baseline B: anyone nearby counts
                d = min((np.linalg.norm(to_floor(feet(r)) - bag_pt) for r in people.itertuples()),
                        default=np.inf)                                          # NEW: floor position

            if d <= dist_limit:
                b["state"], b["timer"] = "ATTENDED", 0
            else:
                b["timer"] += 1
                if b["state"] == "ATTENDED":
                    b["state"] = "UNATTENDED"
                if b["timer"] >= LIMIT and b["state"] != "ABANDONED":
                    b["state"] = "ABANDONED"
                    alerts.append(dict(bag=bid, owner=b["owner"], frame=f, time_s=round(f / fps, 1)))
                    print(f"[{f/fps:5.1f}s] ALERT: bag {bid} abandoned (owner = person {b['owner']})")
        f += 1

    out = Path("results/alerts") / f"{Path(video).stem}_{mode}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(alerts, columns=["bag", "owner", "frame", "time_s"]).to_csv(out, index=False)
    print("saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["owner", "any"], default="owner")
    ap.add_argument("--dist", type=float, default=3.0)   # NEW: meters when calibrated
    ap.add_argument("--time", type=float, default=10)    # seconds
    ap.add_argument("--grace", type=float, default=3)    # seconds
    ap.add_argument("--reid", type=float, default=0.7)   # how closely clothing colors must match (0 to 1)
    a = ap.parse_args()
    run(a.video, a.mode, a.dist, a.time, a.grace, a.reid)