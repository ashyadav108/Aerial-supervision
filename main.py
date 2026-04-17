import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ================= LOAD MODEL =================
model = YOLO("yolov8n.pt")

# ================= IOU =================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

# ================= REMOVE DUPLICATES =================
def remove_duplicates(detections, thresh=0.5):
    final = []
    for det in detections:
        keep = True
        for f in final:
            if iou(det, f) > thresh:
                keep = False
                break
        if keep:
            final.append(det)
    return final

# ================= TILING DETECTION (3x3) =================
def detect(frame):

    h, w, _ = frame.shape
    tile_h, tile_w = h // 3, w // 3

    detections = []

    for i in range(3):
        for j in range(3):

            y1 = i * tile_h
            y2 = (i+1) * tile_h
            x1 = j * tile_w
            x2 = (j+1) * tile_w

            tile = frame[y1:y2, x1:x2]

            results = model(tile, imgsz=960)[0]

            for r in results.boxes:

                x1b, y1b, x2b, y2b = r.xyxy[0].cpu().numpy()
                conf = float(r.conf[0])
                cls = int(r.cls[0])

                # PERSON ONLY
                if cls != 0 or conf < 0.2:
                    continue

                # map back to original frame
                x1b += x1
                x2b += x1
                y1b += y1
                y2b += y1

                detections.append([
                    int(x1b),
                    int(y1b),
                    int(x2b),
                    int(y2b)
                ])

    return remove_duplicates(detections)

# ================= TRACKER =================
class Tracker:

    def __init__(self):

        self.tracks = []
        self.next_id = 0
        self.history = {}

        self.max_history = 25
        self.max_lost = 5

    def update(self, detections):

        updated = []
        assigned = set()

        for det in detections:

            best_iou = 0
            best_idx = -1

            for i, track in enumerate(self.tracks):

                score = iou(track["bbox"], det)

                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_iou > 0.3:

                track = self.tracks[best_idx]
                track["bbox"] = det
                track["lost"] = 0

                updated.append(track)
                assigned.add(best_idx)

            else:

                updated.append({
                    "id": self.next_id,
                    "bbox": det,
                    "lost": 0
                })

                self.next_id += 1

        # handle lost tracks
        for i, track in enumerate(self.tracks):

            if i not in assigned:

                track["lost"] += 1

                if track["lost"] < self.max_lost:
                    updated.append(track)

        self.tracks = updated

        # save history for movement trail
        for t in self.tracks:

            tid = t["id"]

            x1, y1, x2, y2 = t["bbox"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if tid not in self.history:
                self.history[tid] = []

            self.history[tid].append((cx, cy))

            if len(self.history[tid]) > self.max_history:
                self.history[tid].pop(0)

        return self.tracks

# ================= DRAW =================
def draw(frame, tracker, fps):

    for t in tracker.tracks:

        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = t["id"]

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ID label
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Movement trail
        pts = tracker.history.get(tid, [])

        for i in range(1, len(pts)):

            cv2.line(
                frame,
                pts[i-1],
                pts[i],
                (0, 0, 255),
                3
            )

    # ================= FPS DISPLAY =================

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return frame

# ================= MAIN =================

SEQ_DIR = "VisDrone2019-MOT-val/sequences"
OUT_DIR = "outputs/videos"

os.makedirs(OUT_DIR, exist_ok=True)

sequences = sorted(os.listdir(SEQ_DIR))

for seq in sequences:

    print(f"\nProcessing {seq}...")

    seq_path = os.path.join(SEQ_DIR, seq)

    images = sorted(os.listdir(seq_path))

    first_frame = cv2.imread(
        os.path.join(seq_path, images[0])
    )

    if first_frame is None:
        continue

    h, w, _ = first_frame.shape

    out = cv2.VideoWriter(
        os.path.join(OUT_DIR, seq + ".mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (w, h)
    )

    tracker = Tracker()

    prev_time = time.time()
    fps = 0

    start = time.time()

    for img in images:

        if not img.endswith(".jpg"):
            continue

        frame = cv2.imread(
            os.path.join(seq_path, img)
        )

        if frame is None:
            continue

        # ===== FPS CALCULATION =====

        current_time = time.time()

        new_fps = 1 / (current_time - prev_time)

        prev_time = current_time

        fps = 0.9 * fps + 0.1 * new_fps

        # ===== DETECTION =====

        detections = detect(frame)

        # ===== TRACKING =====

        tracker.update(detections)

        # ===== DRAW =====

        frame = draw(frame, tracker, fps)

        out.write(frame)

    end = time.time()

    print("Average FPS:", len(images) / (end - start))

    out.release()

print("\n✅ DONE — Final Improved Output Generated")
