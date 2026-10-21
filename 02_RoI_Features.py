
# C3R — STEP 2: OBJECT-SPECIFIC RoI FEATURES

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from torchvision.transforms.functional import to_tensor
from matplotlib.patches import Rectangle

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# Select top 2 confidently detected objects
selected = torch.where(
    predictions["scores"] >= 0.50
)[0][:2]

if len(selected) == 0:
    raise ValueError("No detections above threshold.")

boxes = predictions["boxes"][selected].detach()
labels = predictions["labels"][selected].detach()
scores = predictions["scores"][selected].detach()

# Extract backbone and aligned RoI features
with torch.no_grad():
    input_tensor = to_tensor(image).to(device)
    images_t, _ = model.transform([input_tensor], None)

    backbone_features = model.backbone(images_t.tensors)

    old_w, old_h = image.size
    new_h, new_w = images_t.image_sizes[0]

    scale = torch.tensor(
        [new_w / old_w, new_h / old_h,
         new_w / old_w, new_h / old_h],
        device=device,
        dtype=boxes.dtype
    )

    transformed_boxes = boxes * scale

    roi_pooled = model.roi_heads.box_roi_pool(
        backbone_features,
        [transformed_boxes],
        images_t.image_sizes
    )

    roi_embeddings = model.roi_heads.box_head(
        roi_pooled
    )

print("Detected objects:", len(selected))
print("RoI pooled shape:", tuple(roi_pooled.shape))
print("RoI embedding shape:", tuple(roi_embeddings.shape))

# Generate object-specific class-conditioned maps
original = np.asarray(image)
H, W = original.shape[:2]

fig, axes = plt.subplots(
    1, len(selected) + 1,
    figsize=(6 * (len(selected) + 1), 5)
)

axes[0].imshow(original)
axes[0].set_title("Clean COCO Detection")
axes[0].axis("off")

for i in range(len(selected)):
    class_id = int(labels[i].item())
    name = categories[class_id]
    confidence = float(scores[i].item())

    # Local gradient at the pooled RoI representation
    roi = (
        roi_pooled[i:i+1]
        .detach()
        .clone()
        .requires_grad_(True)
    )

    with torch.enable_grad():
        emb = model.roi_heads.box_head(roi)

        logits, _ = model.roi_heads.box_predictor(emb)

        target_logit = logits[0, class_id]

        grads = torch.autograd.grad(
            target_logit, roi
        )[0]

        weights_cam = grads.mean(
            dim=(2, 3), keepdim=True
        )

        cam = F.relu(
            (weights_cam * roi).sum(dim=1)[0]
        )

    cam = cam.detach()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    x1, y1, x2, y2 = boxes[i].cpu().tolist()

    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(x1+1, min(W, int(x2)))
    y2 = max(y1+1, min(H, int(y2)))

    patch_h = y2 - y1
    patch_w = x2 - x1

    resized_cam = F.interpolate(
        cam[None, None],
        size=(patch_h, patch_w),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy()

    heatmap = np.zeros((H, W), dtype=np.float32)
    heatmap[y1:y2, x1:x2] = resized_cam

    ax = axes[i+1]
    ax.imshow(original)

    masked_heat = np.ma.masked_where(
        heatmap <= 0, heatmap
    )

    ax.imshow(
        masked_heat,
        cmap="jet",
        alpha=0.55,
        vmin=0,
        vmax=1
    )

    ax.add_patch(
        Rectangle(
            (x1, y1),
            patch_w,
            patch_h,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )
    )

    ax.set_title(
        f"{name} ({confidence:.3f})\n"
        "RoI class-response map"
    )
    ax.axis("off")

    print(
        f"Object {i+1}: {name}, "
        f"confidence={confidence:.4f}"
    )

plt.suptitle(
    "C3R Step 2 — Object-Level RoI Attribution",
    fontsize=15
)
plt.tight_layout()

output_dir = Path(globals().get("C3R_OUTPUT_DIR", "/content/c3r_step1"))
output_dir.mkdir(parents=True, exist_ok=True)

figure_path = output_dir / "step2_roi_response.png"
plt.savefig(
    figure_path, dpi=200,
    bbox_inches="tight"
)
plt.show()

# Save object representations
feature_path = output_dir / "step2_roi_features.npz"

np.savez_compressed(
    feature_path,
    features=roi_embeddings.detach().cpu().numpy(),
    boxes=boxes.cpu().numpy(),
    labels=labels.cpu().numpy(),
    scores=scores.cpu().numpy()
)

print("\nSTEP 2 COMPLETED")
print("Figure:", figure_path)
print("Features:", feature_path)
