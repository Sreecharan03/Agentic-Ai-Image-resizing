import cv2
import numpy as np
import optuna
from PIL import Image, ImageDraw

# === 1. File Paths ===
original_path = r"C:\Users\ASUS\Desktop\Smart Ai\eSBD-package-modifciation-feature-eSBD-modification\eSBD-package-modifciation-feature-eSBD-modification\original.png"
canvas_path   = r"C:\Users\ASUS\Desktop\Smart Ai\eSBD-package-modifciation-feature-eSBD-modification\eSBD-package-modifciation-feature-eSBD-modification\canvas.png"
logo_path     = r"C:\Users\ASUS\Desktop\Smart Ai\eSBD-package-modifciation-feature-eSBD-modification\eSBD-package-modifciation-feature-eSBD-modification\logo.png"
claim_path    = r"C:\Users\ASUS\Desktop\Smart Ai\eSBD-package-modifciation-feature-eSBD-modification\eSBD-package-modifciation-feature-eSBD-modification\claim.png"
output_path   = "final_8.png"
# === 2. Load Base Images ===
original_cv = cv2.imread(original_path)
logo_cv     = cv2.imread(logo_path)
claim_cv    = cv2.imread(claim_path)

if original_cv is None or logo_cv is None or claim_cv is None:
    raise FileNotFoundError("❌ Missing input images. Check file paths!")

canvas_base = Image.open(canvas_path).convert("RGBA")
logo_pil    = Image.open(logo_path).convert("RGBA")
claim_pil   = Image.open(claim_path).convert("RGBA")
W, H = canvas_base.size
canvas_area = W * H

# === 3. Get Ratios from User ===
try:
    base_logo_ratio = float(input("🎯 Enter target brand_logo ratio (e.g., 0.12): "))
    base_claim_ratio = float(input("🎯 Enter target value_claim ratio (e.g., 0.20): "))
except ValueError:
    print("⚠ Invalid input. Using default ratios: logo=0.12, claim=0.20")
    base_logo_ratio = 0.12
    base_claim_ratio = 0.20

# === 4. Template Matching ===
def find_template_coords(base_img, template_img, threshold=0.75):
    result = cv2.matchTemplate(base_img, template_img, cv2.TM_CCOEFF_NORMED)
    yloc, xloc = np.where(result >= threshold)
    rects = []
    ww, hh = template_img.shape[1], template_img.shape[0]
    for (xx, yy) in zip(xloc, yloc):
        rects.append([xx, yy, ww, hh])
        rects.append([xx, yy, ww, hh])
    rects, _ = cv2.groupRectangles(rects, 1, 0.5)
    return rects

logo_boxes  = find_template_coords(original_cv, logo_cv)
claim_boxes = find_template_coords(original_cv, claim_cv)

if len(logo_boxes) == 0 or len(claim_boxes) == 0:
    raise ValueError("❌ Could not detect logo or claim in original image.")

logo_box  = logo_boxes[0]
claim_box = claim_boxes[0]

# === 5. Placement Helpers ===
def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

def resize_by_area(pil_img, ratio, canvas_area):
    orig_area = pil_img.width * pil_img.height
    scale = (ratio * canvas_area / orig_area) ** 0.5
    new_w = int(pil_img.width * scale)
    new_h = int(pil_img.height * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, (new_w, new_h)

def try_local_place(canvas, pil_img, det_box, existing_boxes, ratio, canvas_area, shrink_factor, max_shrink, shift_offsets):
    cx = det_box[0] + det_box[2] // 2
    cy = det_box[1] + det_box[3] // 2
    scale = 1.0

    for shrink in range(max_shrink):
        resized, (w, h) = resize_by_area(pil_img, ratio * (scale ** 2), canvas_area)
        for dx, dy in shift_offsets:
            px = max(0, min(cx + dx - w // 2, W - w))
            py = max(0, min(cy + dy - h // 2, H - h))
            new_box = (px, py, px + w, py + h)
            if all(not boxes_overlap(new_box, b) for b in existing_boxes):
                canvas.alpha_composite(resized, (px, py))
                return new_box
        scale *= shrink_factor
    return None

# === 6. Optuna Objective ===
def objective(trial):
    canvas = canvas_base.copy()
    placed_boxes = []

    logo_ratio  = trial.suggest_float("logo_ratio", max(0.01, base_logo_ratio - 0.01), base_logo_ratio + 0.01)
    claim_ratio = trial.suggest_float("claim_ratio", max(0.01, base_claim_ratio - 0.01), base_claim_ratio + 0.01)
    shrink_factor = trial.suggest_float("shrink_factor", 0.85, 0.95)
    max_shrink = trial.suggest_int("max_shrink", 2, 5)
    shift_range = trial.suggest_int("shift_range", 10, 40)

    shift_offsets = [(dx, dy) for dx in range(-shift_range, shift_range + 1, 10)
                              for dy in range(-shift_range, shift_range + 1, 10)]

    claim_result = try_local_place(canvas, claim_pil, claim_box, placed_boxes,
                                   claim_ratio, canvas_area, shrink_factor, max_shrink, shift_offsets)
    if claim_result is None:
        return 0.0
    placed_boxes.append(claim_result)

    logo_result = try_local_place(canvas, logo_pil, logo_box, placed_boxes,
                                  logo_ratio, canvas_area, shrink_factor, max_shrink, shift_offsets)
    if logo_result is None:
        return 0.0
    placed_boxes.append(logo_result)

    score = sum((x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in placed_boxes)
    return score / canvas_area

# === 7. Optimize ===
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best = study.best_params
print("\n🎯 Best Parameters Found:")
for k, v in best.items():
    print(f"  {k}: {v:.4f}")

# === 8. Final Placement with Best Params ===
canvas = canvas_base.copy()
placed_boxes = []
final_positions = {}

shift_offsets = [(dx, dy) for dx in range(-int(best["shift_range"]), int(best["shift_range"]) + 1, 10)
                            for dy in range(-int(best["shift_range"]), int(best["shift_range"]) + 1, 10)]

claim_result = try_local_place(canvas, claim_pil, claim_box, placed_boxes,
                               best["claim_ratio"], canvas_area, best["shrink_factor"], best["max_shrink"], shift_offsets)
if claim_result:
    placed_boxes.append(claim_result)
    final_positions["value_claim"] = claim_result

logo_result = try_local_place(canvas, logo_pil, logo_box, placed_boxes,
                              best["logo_ratio"], canvas_area, best["shrink_factor"], best["max_shrink"], shift_offsets)
if logo_result:
    placed_boxes.append(logo_result)
    final_positions["brand_logo"] = logo_result

canvas.save(output_path)
print(f"\n✅ Final optimized image saved to: {output_path}")

# === 9. Area Report ===
for label, (x1, y1, x2, y2) in final_positions.items():
    area = (x2 - x1) * (y2 - y1)
    pct = 100 * area / canvas_area
    print(f"📐 {label} final area = {area} px², i.e. {pct:.2f}% of the canvas.")
