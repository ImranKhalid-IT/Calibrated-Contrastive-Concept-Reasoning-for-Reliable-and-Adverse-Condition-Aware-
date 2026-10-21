# C3R STEP 3B — Higher-replication, foreground-restricted region validation.
# Run in the SAME Colab runtime after the uploaded Step 3A file.
# The 120 spatial controls/region (up from 3 in original) are NOT independent samples.

import hashlib
import json
import time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PILImage, ImageFilter

required_3b = ('source', 'image', 'segments', 'x1', 'y1', 'x2', 'y2',
               'record', 'fixed_box_class_logits', 'predicted_class',
               'alternative_class', 'base_margin', 'categories')
missing_3b = [k for k in required_3b if k not in globals()]
if missing_3b:
    raise RuntimeError('Run Step 3A first; missing: ' + ', '.join(missing_3b))

N_CONTROLS_3B = int(globals().get('N_CONTROLS_3B', 120))
MAX_REGIONS_3B = int(globals().get('MAX_REGIONS_3B', 5))
BATCH_3B = int(globals().get('BATCH_3B', 2))
SEED_3B = 42
if N_CONTROLS_3B < 3 or BATCH_3B < 1:
    raise ValueError('N_CONTROLS_3B must be >= 3 and BATCH_3B >= 1.')
source = np.asarray(source, dtype=np.uint8)
H, W = source.shape[:2]
ch, cw = y2-y1, x2-x1
crop = source[y1:y2, x1:x2].copy()

# Estimate object foreground in the local, padded context.
pad = max(12, int(0.30 * max(ch, cw)))
px1, py1 = max(0, x1-pad), max(0, y1-pad)
px2, py2 = min(W, x2+pad), min(H, y2+pad)
context = source[py1:py2, px1:px2].copy()
rx, ry = x1-px1, y1-py1
mask_method = 'GrabCut foreground estimate (not ground truth)'
try:
    cv2.setRNGSeed(SEED_3B)
    gc_mask = np.zeros(context.shape[:2], dtype=np.uint8)
    bg_model, fg_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(context, cv2.COLOR_RGB2BGR), gc_mask,
                (rx, ry, cw, ch), bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    foreground = np.isin(gc_mask[ry:ry+ch, rx:rx+cw], [cv2.GC_FGD, cv2.GC_PR_FGD])
    foreground = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8)).astype(bool)
    if not 0.06 <= foreground.mean() <= 0.94:
        raise ValueError('Uninformative foreground mask')
except (cv2.error, ValueError) as e:
    foreground = np.ones((ch, cw), dtype=bool)
    mask_method = 'Bbox-only fallback (not object mask)'
    print('[Step 3B] WARNING:', str(e))

selected_3b = []
for item in record:
    rid = int(item['region'])
    region_mask = (segments == rid) & foreground
    n = int(region_mask.sum())
    if n >= max(25, int(0.015 * foreground.sum())):
        selected_3b.append((rid, region_mask, n))
    if len(selected_3b) >= MAX_REGIONS_3B:
        break
if not selected_3b:
    raise RuntimeError('No suitable foreground candidates. Inspect mask/segments.')
print('[Step 3B] Foreground:', mask_method)
print('[Step 3B] Selected:', [(r, n) for r, _, n in selected_3b])

rng_3b = np.random.default_rng(SEED_3B)
fg_coords_3b = np.column_stack(np.where(foreground)).astype(np.int32)
blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=14)).convert('RGB'),
                     dtype=np.uint8)[y1:y2, x1:x2]
neutral_color = np.median(crop[foreground], axis=0).astype(np.uint8)
METHODS_3B = ('blur', 'neutral_fill')

def random_control_3b(n, target_mask):
    best, best_overlap = None, 2.0
    for _ in range(40):
        yy, xx = fg_coords_3b[int(rng_3b.integers(len(fg_coords_3b)))].tolist()
        dist = ((fg_coords_3b[:, 0].astype(float)-yy)**2
                + (fg_coords_3b[:, 1].astype(float)-xx)**2)
        indices = np.argpartition(dist, n-1)[:n]
        m = np.zeros((ch, cw), dtype=bool)
        m[fg_coords_3b[indices, 0], fg_coords_3b[indices, 1]] = True
        overlap = float((m & target_mask).sum()/n)
        if overlap < best_overlap:
            best, best_overlap = m, overlap
        if overlap <= .25:
            break
    return best, best_overlap


def edited_image_3b(mask, method):
    edited = source.copy()
    view = edited[y1:y2, x1:x2]
    if method == 'blur':
        view[mask] = blurred[mask]
    elif method == 'neutral_fill':
        view[mask] = neutral_color
    else:
        raise ValueError(method)
    return edited


def logits_batch_3b(arrays):
    if callable(globals().get('fixed_roi_logits_batch_3a')):
        return fixed_roi_logits_batch_3a(arrays)
    return np.stack([fixed_box_class_logits(PILImage.fromarray(a)).detach().float().cpu().numpy()
                     for a in arrays])

# Compose a deterministic list so an interrupted Colab session can resume scoring.
jobs_3b, overlap_3b = [], {}
for rid, m, n in selected_3b:
    for method in METHODS_3B:
        jobs_3b.append(dict(region=rid, mask=m, method=method, kind='observed', trial=-1))
    controls = []
    for trial in range(N_CONTROLS_3B):
        cm, overlap = random_control_3b(n, m)
        controls.append(overlap)
        for method in METHODS_3B:
            jobs_3b.append(dict(region=rid, mask=cm, method=method,
                                kind='control', trial=trial))
    overlap_3b[rid] = controls

out_dir = Path(globals().get('C3R_OUTPUT_DIR', '/content/c3r_step1'))
out_dir.mkdir(parents=True, exist_ok=True)
run_hash_3b = hashlib.sha256(
    source.tobytes() + segments.astype(np.int32).tobytes() + foreground.tobytes()
    + np.asarray(globals().get('base_logits', []), dtype=np.float32).tobytes()
    + json.dumps({'n_controls': N_CONTROLS_3B, 'regions': [int(r) for r,_,_ in selected_3b],
                  'classes': [predicted_class, alternative_class], 'seed': SEED_3B}, sort_keys=True).encode()
).hexdigest()[:16]
progress_path_3b = out_dir/f'step3b_progress_{run_hash_3b}.jsonl'
completed_3b = {}
if progress_path_3b.exists():
    for line in progress_path_3b.read_text().splitlines():
        try:
            entry = json.loads(line)
            completed_3b[int(entry['job_id'])] = entry
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
print(f'[Step 3B] {len(jobs_3b)} edits, {len(completed_3b)} cached; {N_CONTROLS_3B} controls per region')
start_time_3b = time.time()
with progress_path_3b.open('a', encoding='utf-8') as progress_file:
    for k in range(0, len(jobs_3b), BATCH_3B):
        pending = [(i, jobs_3b[i]) for i in range(k, min(k+BATCH_3B, len(jobs_3b)))
                   if i not in completed_3b]
        if pending:
            scores = logits_batch_3b([edited_image_3b(j['mask'], j['method']) for _, j in pending])
            for (index, job), logits in zip(pending, scores):
                margin = float(logits[predicted_class] - logits[alternative_class])
                entry = dict(job_id=index, region=int(job['region']),
                             method=job['method'], kind=job['kind'], trial=int(job['trial']),
                             edited_margin=margin, delta=float(base_margin-margin))
                completed_3b[index] = entry
                progress_file.write(json.dumps(entry)+'\n')
            progress_file.flush()
        if (k//BATCH_3B + 1) % 25 == 0 or k+BATCH_3B >= len(jobs_3b):
            print(f'[Step 3B] {min(k+BATCH_3B,len(jobs_3b))}/{len(jobs_3b)} jobs checked'
                  f' ({time.time()-start_time_3b:.0f}s elapsed)')

records_3b = [completed_3b[i] for i in range(len(jobs_3b))]
results_3b = []
for rid, region_mask, n in selected_3b:
    measurements = {}
    for method in METHODS_3B:
        rows = [r for r in records_3b if r['region']==rid and r['method']==method]
        observed = next(r['delta'] for r in rows if r['kind']=='observed')
        controls = np.asarray([r['delta'] for r in rows if r['kind']=='control'], dtype=float)
        abs_controls = np.abs(controls)
        measurements[method] = {
            'effect': float(observed), 'control_values': controls.tolist(),
            'control_abs_max': float(abs_controls.max()),
            'q95_control_abs': float(np.quantile(abs_controls, .95)),
            'outside_control_q95': bool(abs(observed) > np.quantile(abs_controls, .95)),
            'outside_sampled_control_range': bool(abs(observed) > abs_controls.max()),
            # Descriptive Monte Carlo exceedance; spatial controls are correlated.
            'absolute_control_exceedance_fraction': float((1+(abs_controls >= abs(observed)).sum())/(len(controls)+1)),
        }
    results_3b.append({
        'region': rid, 'pixels': n, 'median_control_overlap': float(np.median(overlap_3b[rid])),
        'measurements': measurements,
    })
    print('[Step 3B] Region', rid,
          '| blur', round(measurements['blur']['effect'], 3),
          '| fill', round(measurements['neutral_fill']['effect'], 3),
          '| median overlap', round(float(np.median(overlap_3b[rid])), 3))

fig, axs = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
axs[0, 0].imshow(source)
axs[0, 0].add_patch(Rectangle((x1, y1), cw, ch, fill=False, edgecolor='lime', linewidth=2))
axs[0, 0].set_title('A. Clean image / fixed target box')
axs[0, 1].imshow(crop)
axs[0, 1].imshow(np.ma.masked_where(~foreground, foreground), cmap='Greens', alpha=.45)
axs[0, 1].set_title('B. Foreground proxy (not ground truth)')
colored = crop.copy()
for idx, (rid, m, _) in enumerate(selected_3b):
    color = np.array(plt.get_cmap('tab10')(idx)[:3]) * 255
    colored[m] = np.asarray(.5*colored[m] + .5*color, dtype=np.uint8)
axs[0, 2].imshow(colored)
axs[0, 2].set_title(f'C. {len(selected_3b)} candidate regions')
for ax, (rid, m, n) in zip(axs[1, :2], selected_3b):
    ax.imshow(crop)
    ax.imshow(np.ma.masked_where(~m, m), cmap='autumn', alpha=.50)
    ax.set_title(f'Region {rid} ({n} pixels)')
if len(selected_3b) == 1:
    axs[1, 1].imshow(crop)
    axs[1, 1].set_title('D/E. One region qualified')

x_positions = np.arange(len(results_3b))
bar_width = .36
for j, method in enumerate(METHODS_3B):
    signed = [r['measurements'][method]['effect'] for r in results_3b]
    quantile = [r['measurements'][method]['q95_control_abs'] for r in results_3b]
    centers = x_positions + (j-.5)*bar_width
    axs[1, 2].bar(centers, signed, bar_width, label=method)
    axs[1, 2].scatter(centers, quantile, marker='^', color='black', s=25)
    axs[1, 2].scatter(centers, [-v for v in quantile], marker='v', color='black', s=25)
axs[1, 2].axhline(0, color='black', linewidth=.8)
axs[1, 2].set_xticks(x_positions, [f"R{r['region']}" for r in results_3b])
axs[1, 2].set_ylabel('Change in contrastive logit margin')
axs[1, 2].set_title('F. Effects vs |control| 95th percentile')
axs[1, 2].legend(fontsize=8)
for ax in axs.flat:
    if ax is not axs[1, 2]:
        ax.axis('off')
fig.suptitle(f'C3R Step 3B | {categories[predicted_class]} vs {categories[alternative_class]}\n'
             'Expanded descriptive controls; no semantic or causal validation', fontsize=13)
figure_path = out_dir/'step3b_results.png'
json_path = out_dir/'step3b_results.json'
fig.savefig(figure_path, dpi=190, bbox_inches='tight')
plt.show()
with open(json_path, 'w', encoding='utf-8') as output:
    json.dump({
        'target_class': str(categories[predicted_class]),
        'alternative_class': str(categories[alternative_class]),
        'baseline_margin': float(base_margin),
        'foreground_method': mask_method,
        'n_controls': N_CONTROLS_3B,
        'n_regions': len(selected_3b),
        'regions': results_3b,
        'calibrated_reliability': None,
        'semantic_roles_validated': False,
        'cautions': ['Spatial controls are correlated; descriptive threshold, not statistical significance',
                     'More control samples improve stability of quantile estimates but do not train the detector'],
    }, output, indent=2)
print('[Step 3B] Figure:', figure_path)
print('[Step 3B] Data:', json_path)
print('[Step 3B] Resume checkpoint:', progress_path_3b)
print('STEP 3B COMPLETED — now run Step 3C.')
