import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

from configs.config import CONF_THRESHOLD, OUTPUT_DIR, RAW_DATA_DIR, SAVE_VISUALIZATION, YOLO_BEST_MODEL


# =========================
# 📁 INIT
# =========================
input_dir = Path(RAW_DATA_DIR)
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

vis_dir = output_dir / "visualizations"
vis_dir.mkdir(exist_ok=True)

# CSV путь
csv_path = output_dir / "results.csv"


# =========================
# 🤖 MODEL
# =========================
model = YOLO(YOLO_BEST_MODEL)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# =========================
# 🔄 PROCESSING
# =========================
rows = []

image_paths = list(input_dir.glob("*.*"))

for img_path in tqdm(image_paths):
    image = cv2.imread(str(img_path))

    if image is None:
        print(f"⚠️ Ошибка чтения: {img_path}")
        continue

    # инференс
    results = model(image)[0]

    # в supervision
    detections = sv.Detections.from_ultralytics(results)

    # фильтр по confidence
    if len(detections) > 0:
        detections = detections[detections.confidence > CONF_THRESHOLD]

    has_plate = len(detections) > 0

    # =========================
    # 🖼 ВИЗУАЛИЗАЦИЯ
    # =========================
    if SAVE_VISUALIZATION:
        annotated = image.copy()

        if has_plate:
            labels = [
                f"{conf:.2f}"
                for conf in detections.confidence
            ]

            annotated = box_annotator.annotate(
                scene=annotated,
                detections=detections
            )

            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels
            )

        # если нет таблички — просто сохраняем как есть
        cv2.imwrite(str(vis_dir / img_path.name), annotated)

    # =========================
    # 📊 ЛОГИ
    # =========================
    rows.append({
        "image": img_path.name,
        "has_plate": has_plate,
        "detections_count": len(detections)
    })


# =========================
# 💾 SAVE CSV
# =========================
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

print("\n✅ Готово!")
print(f"CSV: {csv_path}")
print(f"Визуализации: {vis_dir}")