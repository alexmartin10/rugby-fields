from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

model.predict(
    source="data/yolo_dataset/v1/images/test",
    conf=0.25,
    save=True,
)