import torch
from ultralytics import YOLO
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap = cv2.VideoCapture("video/A1606b0e6_0.mp4")
model = YOLO('model/best.pt').to(device)
w, h = int(cap.get(3)), int(cap.get(4))
if not cap.isOpened():
    exit(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w//3, h//3))
    if not ret:
        break
    resulst = model.predict(frame, show = True)[0]

    for r in resulst:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            conf = float(box.conf)
            clsid = int(box.cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "{} {}".format(model.names[clsid], conf)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_ITALIC, 0.6, (0, 255, 0), 2)
    cv2.imshow("dsa",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()