from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('runs/detect/train11/weights/best.pt')
results = model.predict(r"Image_Path", save=True, imgsz=320, conf=0.2)

image_path = r"Image_Path"
image = cv2.imread(image_path)

for i, result in enumerate(results):
    boxes = result.boxes.xyxy
    for j, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = image[ymin:ymax, xmin:xmax]

        plt.figure()
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Cropped Image {i}_{j}')
        plt.axis('off')
        plt.show()
