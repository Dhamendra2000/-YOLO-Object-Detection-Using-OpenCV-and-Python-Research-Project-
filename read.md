#This is the project for visually impared person 
#our teams are working on one prototype model which will help to visually impared person 
#this is the small segement of that project and which has been done by me 


# Blind-Aware Dynamic ROI Object Detection using YOLOv8:-

A real-time object detection system with **Dynamic Region of Interest (ROI)** logic using **YOLOv8**, designed to simulate smart navigation systems — such as assistive technology for the visually impaired.

---

# Objective:-

- Perform object detection on video streams using YOLOv8.
- Define and dynamically shift an ROI (Region of Interest) to simulate smart navigation.
- Evaluate detection accuracy using dummy ground truth labels.
- Save output frames optionally for use in academic research or publications.

---

#Technologies Used:-

| Tool / Library     | Purpose                         |
|--------------------|----------------------------------|
| Python             | Programming language             |
| OpenCV             | Video and image processing       |
| Ultralytics YOLOv8 | Real-time object detection       |
| NumPy              | Array and mathematical operations |

---

# Features:-

-  Real-time object detection using YOLOv8.
- Dynamic ROI that adjusts based on object positions.
- Evaluation metrics: Precision, Recall, F1 Score, Accuracy.
- Frame saving option for research visuals.
- Visualization of detections and ROI on video frames.

---

# Project Structure :-

```plaintext
# project-folder/
│
├──  main.py                # Main detection + evaluation script
├──  sample_video.mp4       # Input video file
├──  yolov8n.pt              # YOLOv8 model (update path if needed)
├──  output_frames/         # (Optional) Saved annotated frames
├──  README.md              # Project documentation

