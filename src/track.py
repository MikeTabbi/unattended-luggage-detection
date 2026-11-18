import argparse, csv
from pathlib import Path
from ultralytics import YOLO

CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    
    # low confidence threshold, so faint bags are kept
    ap.add_argument("--conf", type=float, default=0.1)
    args = ap.parse_args()

    out = Path("results/tracks") / (Path(args.video).stem + ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    stream = model.track(source=args.video, classes=list(CLASSES), conf=args.conf,
                         imgsz=args.imgsz,
                         tracker="bytetrack.yaml", device="mps",
                         stream=True, persist=True, verbose=False)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2"])
        for frame_idx, r in enumerate(stream):
            b = r.boxes
            
            # Process each detected object in the frame
            if b.id is None:
                continue
            for tid, c, cf, box in zip(b.id.int().tolist(), b.cls.int().tolist(),
                                       b.conf.tolist(), b.xyxy.tolist()):
                w.writerow([frame_idx, tid, CLASSES[c], round(cf, 3),
                            *[round(v, 1) for v in box]])
    print("saved", out)

if __name__ == "__main__":
    main()