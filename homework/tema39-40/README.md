Tema 39-40 – YOLO Real-Time Object Detection

Task I
A real-time object detection system was implemented using a live IP camera stream
captured from a smartphone. Frames were processed using YOLOv5 and displayed
with bounding boxes and confidence scores.

Task II
Three models were benchmarked: YOLOv5m, YOLOv5l and YOLOv5x.
All tests were performed using CPU inference with fixed input resolution (416x416)
and identical environmental conditions.

Results:

YOLOv5m → Highest FPS (~5.5), most detections, lower confidence.
YOLOv5l → Balanced performance, higher confidence (~0.83).
YOLOv5x → Lowest FPS (~1.5), highest computational cost.

Conclusion:
YOLOv5m is most suitable for real-time CPU deployment,
while YOLOv5l provides better detection reliability.

Task III
Frames containing detected objects were automatically saved using timestamped
filenames for later qualitative analysis.