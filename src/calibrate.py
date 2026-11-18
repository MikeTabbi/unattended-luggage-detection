import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

WALK_SPEED = 1.4   # typical adult walking speed, meters per second

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--points", required=True,
                    help="4 corners as x1,y1,x2,y2,...: bottom-left, bottom-right, top-right, top-left")
    ap.add_argument("--square-m", type=float,
                    help="size of the square in meters; overrides the walking-speed estimate")
    a = ap.parse_args()

    px = np.array([float(v) for v in a.points.split(",")], np.float32).reshape(4, 2)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)   # the real shape: a 1 x 1 square
    H, _ = cv2.findHomography(px, square)

    # Measure walking speed on the floor map, in square-widths per second
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS)
    step = int(fps / 2)                                               # compare positions 0.5 s apart
    df = pd.read_csv(Path("results/tracks") / (Path(a.video).stem + ".csv"))
    people = df[df.cls == "person"].copy()
    feet = np.stack([(people.x1 + people.x2) / 2, people.y2], axis=1).astype(np.float32)
    floor = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
    people["fx"], people["fy"] = floor[:, 0], floor[:, 1]

    speeds = []
    for _, t in people.groupby("track_id"):
        t = t.set_index("frame")
        later = t.reindex(t.index + step)                             # same person 0.5 s later
        d = np.hypot(later.fx.values - t.fx.values, later.fy.values - t.fy.values)
        speeds += list(d[~np.isnan(d)] / 0.5)
    speeds = np.array(speeds)
    moving = speeds[speeds > 0.3 * np.median(speeds)]                 # ignore standing still
    unit_speed = float(np.median(moving))
    m_per_unit = WALK_SPEED / unit_speed
    if a.square_m:
        m_per_unit = a.square_m
    print(f"implied walking speed: {unit_speed * m_per_unit:.2f} m/s")

    H_m = np.diag([m_per_unit, m_per_unit, 1.0]) @ H                  # pixels -> meters
    out = Path("calib") / (Path(a.video).stem + ".json")
    out.parent.mkdir(exist_ok=True)
    json.dump({"points_px": px.tolist(), "m_per_unit": m_per_unit, "H": H_m.tolist()},
              open(out, "w"), indent=2)

    print(f"walking speed on map: {unit_speed:.2f} squares/s")
    print(f"scale: 1 square = {m_per_unit:.2f} m  (sanity check: a manhole frame is about 0.7-1.2 m)")
    print("saved", out)

if __name__ == "__main__":
    main()