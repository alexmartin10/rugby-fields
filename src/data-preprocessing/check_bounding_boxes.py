from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

IMAGES_DIR = Path("images")
LABELS_DIR = Path("labels")

# Optionnel : noms des classes
CLASS_NAMES = {
    0: "field"
}

def draw_yolo_boxes(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls_id, x_center, y_center, box_w, box_h = map(float, line.split())

            cls_id = int(cls_id)

            # YOLO normalized -> pixels
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = CLASS_NAMES.get(cls_id, str(cls_id))
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    return image

def show_random_samples(n=6):
    image_paths = list(IMAGES_DIR.glob("*.jpg"))

    samples = random.sample(image_paths, min(n, len(image_paths)))

    for image_path in samples:
        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        image = draw_yolo_boxes(image_path, label_path)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")
        plt.show()

show_random_samples(n=6)