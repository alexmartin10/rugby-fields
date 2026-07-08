from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data = "configs/dataset_v1.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    patience=10,
    workers=4,
)