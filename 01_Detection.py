
# C3R — STEP 1: REAL COCO OBJECT DETECTION

import torch
import torchvision

# TorchVision compatibility
if not hasattr(torchvision, "_is_tracing"):
    torchvision._is_tracing = torch.jit.is_tracing

from pathlib import Path
import zipfile
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

# 1. Download real MS COCO images
root = Path(globals().get("C3R_OUTPUT_DIR", "/content/c3r_step1"))
images_dir = root / "coco128/images/train2017"

if not images_dir.exists() or not list(images_dir.glob("*.jpg")):
    print("Downloading COCO128...")
    root.mkdir(parents=True, exist_ok=True)

    url = (
        "https://github.com/ultralytics/assets/"
        "releases/download/v0.0.0/coco128.zip"
    )

    archive = root / "coco128.zip"

    with requests.get(url, stream=True, timeout=90) as response:
        response.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(archive) as z:
        z.extractall(root)

image_files = sorted(images_dir.glob("*.jpg"))
assert image_files, "COCO128 images not found."

preferred = images_dir / "000000000009.jpg"
image_path = preferred if preferred.exists() else image_files[0]
image = Image.open(image_path).convert("RGB")

print("Selected COCO image:", image_path.name)

# 2. Load pretrained frozen Faster R-CNN
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights,
    min_size=480,
    max_size=720
)

model = model.to(device).eval()

for param in model.parameters():
    param.requires_grad_(False)

print("Model: Faster R-CNN ResNet50 FPN")
print("Device:", device)
print("Detector: FROZEN")

# 3. Run real inference
with torch.inference_mode():
    predictions = model([
        to_tensor(image).to(device)
    ])[0]

categories = weights.meta["categories"]

# 4. Visualize detected objects
fig, ax = plt.subplots(figsize=(11, 7))
ax.imshow(image)

detected_count = 0

for box, score, label in zip(
    predictions["boxes"].cpu(),
    predictions["scores"].cpu(),
    predictions["labels"].cpu()
):
    if float(score) < 0.50 or detected_count >= 8:
        break

    x1, y1, x2, y2 = box.tolist()
    class_name = categories[int(label)]

    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )
    )

    ax.text(
        x1,
        max(y1-4, 0),
        f"{class_name}: {float(score):.3f}",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="green", alpha=0.8, pad=2)
    )

    detected_count += 1

ax.set_title(
    "C3R Step 1 — Frozen Faster R-CNN / MS COCO",
    fontsize=14
)
ax.axis("off")
plt.tight_layout()

output_path = root / "step1_clean_detection.png"
plt.savefig(output_path, dpi=170, bbox_inches="tight")
plt.show()

print("Detected objects:", detected_count)
print("Saved image:", output_path)
print("STEP 1 COMPLETED")
