from ultralytics import YOLO

model = YOLO("models/yolo26n/v1_1024_100e/best.pt")

metrics = model.val(
    data="configs/dataset_external_gironde.yaml",
    split="test",
    imgsz=1024,
)

print(metrics)