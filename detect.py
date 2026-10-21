import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

WEIGHTS = Path("pothole_best.pt")
CONF = 0.10                      # Colab me jo value use ki thi wahi rakho
LABEL_SCALE, LABEL_THICK = 0.6, 1
OUT_DIR = Path("predictions"); OUT_DIR.mkdir(exist_ok=True)

if not WEIGHTS.exists():
    sys.exit(f"Weights nahi mili: {WEIGHTS.resolve()}")
paths = [Path(a) for a in sys.argv[1:]]
if not paths:
    sys.exit("Use: python detect.py road1.jpg [road2.png ...]")

model = YOLO(str(WEIGHTS))
shown = 0
for p in paths:
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        print(f"Skipped (sirf JPG/JPEG/PNG): {p}"); continue
    img = cv2.imread(str(p))
    if img is None:
        print(f"Image not detected : {p}"); continue

    t = max(2, max(img.shape[:2]) // 400)
    boxes = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0].boxes
    for (x1, y1, x2, y2), c, k in zip(boxes.xyxy.int().tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), t)
        cv2.putText(img, f"{model.names[k]} {c:.2f}", (x1, max(y1 - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 0, 255), LABEL_THICK)

    if len(boxes) == 0:
        cv2.putText(img, "No pothole detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.2, (0, 0, 255), LABEL_THICK)
        print(f"{p.name}: No pothole detected (ye prove nahi karta ki road pothole-free hai)")
    else:
        print(f"{p.name}: {len(boxes)} pothole(s) detected")

    out = OUT_DIR / f"{p.stem}_pred.jpg"
    cv2.imwrite(str(out), img)
    print("Saved:", out)

    h, w = img.shape[:2]
    s = min(1000 / w, 700 / h, 1)                      # bade image ko screen me fit karne ke liye
    cv2.namedWindow(p.name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(p.name, int(w * s), int(h * s))
    cv2.imshow(p.name, img); shown += 1

if shown:
    print("Press any key to stop.")
    cv2.waitKey(0); cv2.destroyAllWindows()