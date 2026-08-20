from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
)
from ultralytics import YOLO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "runs/detect/training/runs/yolov8_drowsiness/weights/best.pt"
TEST_IMAGES = "training/datasets/Yolo/combined/test/images"

OUTPUT_DIR = Path("runs/evaluation/roc_auc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Class mapping:
# 0 = alert
# 1 = drowsy


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = YOLO(MODEL_PATH)

print("=" * 60)
print("ROC / AUC EVALUATION")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Test images: {TEST_IMAGES}")
print(f"Classes: {model.names}")


# --------------------------------------------------
# COLLECT GROUND TRUTH + DROWSY SCORES
# --------------------------------------------------

y_true = []
y_score = []

test_dir = Path(TEST_IMAGES)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = sorted(
    [p for p in test_dir.iterdir() if p.suffix.lower() in image_extensions]
)

print(f"\nTest images found: {len(image_files)}")


for index, image_path in enumerate(image_files, start=1):

    # Corresponding YOLO label
    label_path = (
        image_path.parent.parent
        / "labels"
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        print(f"WARNING: Missing label: {label_path}")
        continue

    # Read ground-truth class
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"WARNING: Empty label: {label_path}")
        continue

    # Each test image contains one annotated instance
    gt_class = int(lines[0].split()[0])

    # Drowsy = positive class
    y_true.append(1 if gt_class == 1 else 0)

    # Run YOLO
    result = model.predict(
        source=str(image_path),
        imgsz=640,
        device=0,
        verbose=False,
    )[0]

    drowsy_confidence = 0.0

    if result.boxes is not None and len(result.boxes) > 0:

        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for cls, conf in zip(classes, confidences):

            if int(cls) == 1:  # Drowsy
                drowsy_confidence = max(
                    drowsy_confidence,
                    float(conf)
                )

    y_score.append(drowsy_confidence)

    if index % 50 == 0 or index == len(image_files):
        print(f"Processed {index}/{len(image_files)}")


# --------------------------------------------------
# ROC / AUC
# --------------------------------------------------

auc = roc_auc_score(y_true, y_score)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)


# --------------------------------------------------
# SAVE ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"YOLOv8n (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - YOLOv8n Drowsiness Detection")
plt.legend()
plt.grid(True)

roc_path = OUTPUT_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# SAVE NUMERICAL RESULTS
# --------------------------------------------------

results_path = OUTPUT_DIR / "roc_auc_results.txt"

with open(results_path, "w") as f:

    f.write("YOLOv8n ROC/AUC Evaluation\n")
    f.write("=" * 40 + "\n")
    f.write(f"Test images evaluated: {len(y_true)}\n")
    f.write("Class 0: Alert\n")
    f.write("Class 1: Drowsy\n")
    f.write(f"AUC: {auc:.6f}\n")


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("ROC / AUC RESULTS")
print("=" * 60)

print(f"Images evaluated : {len(y_true)}")
print(f"AUC              : {auc:.6f}")

print(f"\nROC curve saved to:")
print(roc_path)

print(f"\nNumerical results saved to:")
print(results_path)

print("=" * 60)