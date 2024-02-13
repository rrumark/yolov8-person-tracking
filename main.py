import cv2
import numpy as np
import math
from ultralytics import YOLO


cap = cv2.VideoCapture("data/9.mp4")

model = YOLO("yolo-Weights/yolov8n.pt")

frame_count = 0
frame_skip = 25

while True:
    success, frame = cap.read()
    if not success:
        break

    cv2.namedWindow("Webcam")

    if frame_count % frame_skip == 0:
        results = model.track(
            frame, conf=0.3, iou=0.5, persist=True, show=False, verbose=False
        )

        for r in results:
            boxes = r.boxes

            for box in boxes:
                cls = int(box.cls[0])

                if cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )

                    person_id = box.id.item()

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(
                        frame,
                        str(person_id),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
