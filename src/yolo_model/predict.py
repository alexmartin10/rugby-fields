from ultralytics import YOLO

model = YOLO("models/yolo26n-obb/v4_1024_100e/weights/best.pt")

results = model.predict(
    source="/home/alexandre-martin/Documents/rugby-fields/data/raw/D33/dalle_jpg/34.jpg",
    conf=0.001,
    save=True,
)

for r in results:
    print(r.obb.conf)