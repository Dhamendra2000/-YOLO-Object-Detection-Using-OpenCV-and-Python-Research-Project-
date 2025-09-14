import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------- Configuration ----------
MODEL_PATH = "yolov8n - Copy.pt"  # Update if path changes
VIDEO_PATH = "sample_video.mp4"
SAVE_FRAMES = False  # Set True to save frames for research paper
FRAME_SAVE_DIR = "output_frames"

# ---------- Load Model ----------
model = YOLO(MODEL_PATH)

# ---------- Load Video ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Error: Could not open video.")

# ---------- Dummy Ground Truths (for illustration only) ----------
ground_truth_boxes = defaultdict(list)
# Example: ground_truth_boxes[0].append(("person", [x1, y1, x2, y2]))

# ---------- Helper Functions ----------
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def draw_roi(frame, width, height, offset):
    roi_vertices = np.array([[(
        int(width * 0.05) + offset, height),
        (int(width * 0.35) + offset, int(height * 0.5)),
        (int(width * 0.65) + offset, int(height * 0.5)),
        (int(width * 0.95) + offset, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, roi_vertices, (255, 255, 255))
    cv2.polylines(frame, roi_vertices, isClosed=True, color=(0, 255, 0), thickness=2)
    return roi_vertices

# ---------- Evaluation Stats ----------
TP = FP = FN = 0
frame_count = 0
offset = 0
max_offset = 100

# ---------- Frame-by-frame Processing ----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    roi_vertices = draw_roi(frame, width, height, offset)

    # Object Detection
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((label, [x1, y1, x2, y2]))

        # Check if detection is within ROI
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(roi_vertices[0], (cx, cy), False) >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

    # --- Evaluation Logic ---
    matched_gt = set()
    matched_det = set()
    gt_objs = ground_truth_boxes[frame_count]

    for i, (det_label, det_box) in enumerate(detections):
        matched = False
        for j, (gt_label, gt_box) in enumerate(gt_objs):
            if j in matched_gt:
                continue
            if det_label == gt_label and compute_iou(det_box, gt_box) > 0.5:
                TP += 1
                matched_gt.add(j)
                matched_det.add(i)
                matched = True
                break
        if not matched:
            FP += 1

    FN += len(gt_objs) - len(matched_gt)

    # --- ROI Shift Logic (Smart Navigation) ---
    shift_left = shift_right = False
    for _, box in detections:
        cx = (box[0] + box[2]) // 2
        left_x = roi_vertices[0][1][0]
        right_x = roi_vertices[0][2][0]
        if cx < left_x + 20:
            shift_left = True
        elif cx > right_x - 20:
            shift_right = True

    shift_step = 2
    if shift_left:
        offset = max(-max_offset, offset - shift_step)
    elif shift_right:
        offset = min(max_offset, offset + shift_step)
    else:
        offset -= 1 if offset > 0 else -1 if offset < 0 else 0

    # Save frames (optional for research visuals)
    if SAVE_FRAMES:
        cv2.imwrite(f"{FRAME_SAVE_DIR}/frame_{frame_count:04d}.jpg", frame)

    # Display output
    cv2.imshow("Blind-Aware Dynamic ROI Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# ---------- Release and Cleanup ----------
cap.release()
cv2.destroyAllWindows()

# ---------- Final Metrics ----------
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

print("\n--- Evaluation Metrics ---")
print(f"Frames Processed: {frame_count}")
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
