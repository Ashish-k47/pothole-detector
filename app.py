"""Pothole Detector - simple desktop UI (Tkinter + Ultralytics YOLO).

Run:  python app.py      (keep pothole_best.pt in the same folder as this file)
"""
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

WEIGHTS = Path(__file__).with_name("pothole_best.pt")
DEFAULT_CONF = 0.10                 # same idea as CONF in the Colab notebook
LABEL_SCALE, LABEL_THICK = 0.6, 1   # label text size / weight on the red boxes
PANEL_W, PANEL_H = 520, 400
IMG_TYPES = [("Images", "*.jpg *.jpeg *.png")]


def read_image(path):
    """cv2.imread fails on non-English Windows paths, so decode from bytes."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if not ok:
        raise IOError("Could not encode the image.")
    buf.tofile(path)


def to_tk(bgr):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pil.thumbnail((PANEL_W, PANEL_H))
    return ImageTk.PhotoImage(pil)


def annotate(img, boxes, names):
    out = img.copy()
    t = max(2, max(out.shape[:2]) // 400)
    for (x1, y1, x2, y2), c, k in zip(boxes.xyxy.int().tolist(), boxes.conf.tolist(), boxes.cls.int().tolist()):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), t)   # red (BGR)
        cv2.putText(out, f"{names[k]} {c:.2f}", (x1, max(y1 - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, (0, 0, 255), LABEL_THICK)
    if len(boxes) == 0:
        cv2.putText(out, "No pothole detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE * 1.2, (0, 0, 255), LABEL_THICK)
    return out


class App:
    def __init__(self, root):
        self.root = root
        root.title("Pothole Detector")
        root.resizable(False, False)

        self.model = None
        self.path = None      # selected image path
        self.img = None       # original image (BGR)
        self.result = None    # annotated image (BGR)
        self.busy = False
        self.q = queue.Queue()
        self._photos = {}     # keep references so Tk doesn't garbage-collect images

        self.build_ui()
        self.root.after(100, self.poll)

        if not WEIGHTS.exists():
            self.status.config(text=f"Weights not found: {WEIGHTS}", fg="red")
        else:
            self.status.config(text="Loading model...", fg="black")
            threading.Thread(target=self.load_model, daemon=True).start()
        self.refresh_buttons()

    # ---------- UI ----------
    def build_ui(self):
        bar = ttk.Frame(self.root, padding=(10, 10, 10, 4))
        bar.pack(fill="x")
        self.btn_choose = ttk.Button(bar, text="📂  Choose Image", command=self.choose)
        self.btn_clear = ttk.Button(bar, text="🗑  Clear", command=self.clear)
        self.btn_save = ttk.Button(bar, text="💾  Save Result", command=self.save)
        for b in (self.btn_choose, self.btn_clear, self.btn_save):
            b.pack(side="left", padx=(0, 8))

        self.conf = tk.DoubleVar(value=DEFAULT_CONF)
        self.conf_text = ttk.Label(bar, text=f"{DEFAULT_CONF:.2f}", width=5)
        self.conf_text.pack(side="right")
        self.scale = ttk.Scale(bar, from_=0.05, to=0.90, variable=self.conf, length=200,
                               command=lambda _v: self.conf_text.config(text=f"{self.conf.get():.2f}"))
        self.scale.pack(side="right", padx=6)
        self.scale.bind("<ButtonRelease-1>", lambda _e: self.detect())
        ttk.Label(bar, text="Confidence threshold:").pack(side="right")

        panels = ttk.Frame(self.root, padding=10)
        panels.pack()
        self.orig_lbl = self.make_panel(panels, "Original", "No image selected")
        self.res_lbl = self.make_panel(panels, "Detection result", "Result will appear here")
        self.placeholders = {self.orig_lbl: "No image selected", self.res_lbl: "Result will appear here"}

        self.status = tk.Label(self.root, text="", anchor="w", justify="left", wraplength=1040, padx=12, pady=8)
        self.status.pack(fill="x")

    def make_panel(self, parent, title, placeholder):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.pack(side="left", padx=6)
        frame = tk.Frame(box, width=PANEL_W, height=PANEL_H, bg="#ececec")
        frame.pack_propagate(False)
        frame.pack()
        lbl = tk.Label(frame, text=placeholder, bg="#ececec", fg="gray")
        lbl.pack(expand=True, fill="both")
        return lbl

    def set_panel(self, lbl, bgr):
        if bgr is None:
            lbl.config(image="", text=self.placeholders[lbl])
            self._photos.pop(lbl, None)
        else:
            self._photos[lbl] = to_tk(bgr)
            lbl.config(image=self._photos[lbl], text="")

    def refresh_buttons(self):
        ready = self.model is not None and not self.busy
        self.btn_choose.config(state="normal" if ready else "disabled")
        self.btn_clear.config(state="normal" if self.img is not None and not self.busy else "disabled")
        self.btn_save.config(state="normal" if self.result is not None and not self.busy else "disabled")

    # ---------- actions ----------
    def choose(self):
        path = filedialog.askopenfilename(title="Select a road image", filetypes=IMG_TYPES)
        if not path:
            return
        img = read_image(path)
        if img is None:
            messagebox.showerror("Error", "Could not read this image. Use a JPG, JPEG or PNG file.")
            return
        self.path, self.img, self.result = Path(path), img, None
        self.set_panel(self.orig_lbl, img)
        self.set_panel(self.res_lbl, None)
        self.detect()

    def clear(self):
        self.path = self.img = self.result = None
        self.set_panel(self.orig_lbl, None)
        self.set_panel(self.res_lbl, None)
        self.status.config(text="Cleared. Choose another image.", fg="black")
        self.refresh_buttons()

    def save(self):
        if self.result is None:
            return
        out = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=f"{self.path.stem}_pred.jpg",
                                           filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not out:
            return
        try:
            write_image(out, self.result)
            self.status.config(text=f"Saved: {out}", fg="black")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def detect(self):
        if self.img is None or self.model is None or self.busy:
            return
        self.busy = True
        self.refresh_buttons()
        self.status.config(text="Detecting...", fg="black")
        threading.Thread(target=self.worker, args=(self.img, self.conf.get()), daemon=True).start()

    # ---------- background work (results come back through the queue) ----------
    def load_model(self):
        try:
            self.q.put(("model", YOLO(str(WEIGHTS))))
        except Exception as e:
            self.q.put(("error", f"Could not load model: {e}"))

    def worker(self, img, conf):
        try:
            boxes = self.model.predict(img, conf=conf, imgsz=640, verbose=False)[0].boxes
            self.q.put(("result", (annotate(img, boxes, self.model.names), boxes.conf.tolist(), conf)))
        except Exception as e:
            self.q.put(("error", f"Detection failed: {e}"))

    def poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "model":
                    self.model = data
                    self.status.config(text="Model ready. Click 'Choose Image' to start.", fg="black")
                elif kind == "result":
                    out, scores, conf = data
                    self.result = out
                    self.set_panel(self.res_lbl, out)
                    if scores:
                        self.status.config(fg="black", text=f"{len(scores)} pothole(s) detected | confidence: "
                                           + ", ".join(f"{s:.2f}" for s in scores))
                    else:
                        self.status.config(fg="black", text=f"No pothole detected at confidence >= {conf:.2f}. "
                                           "This does not prove the road is pothole-free; try a lower threshold.")
                    self.busy = False
                else:  # error
                    self.status.config(text=data, fg="red")
                    self.busy = False
                self.refresh_buttons()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
