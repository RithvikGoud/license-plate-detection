from ultralytics import YOLO

model = YOLO('C:/Users/MRITH/Downloads/Number-Plate-Recognition/runs/detect/train11/weights/best.pt')
model.predict(r"C:\Users\MRITH\OneDrive\Documents\DL\CA\test\pq.jpg", save=True, imgsz=320, conf=0.5)

