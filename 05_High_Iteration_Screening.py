# C3R STEP 3C -- controlled foreground-restricted intervention screening
# Paste into ONE NEW CELL after 3B. Uses the SAME frozen Faster R-CNN.
# IMPORTANT: this does NOT establish semantic concepts, causal identification, or calibrated reliability.

import gc, json, time, hashlib, csv
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PILImage, ImageFilter
from torchvision.transforms.functional import to_tensor

required_3c = ['model','device','image','source','segments','foreground','record',
               'x1','y1','x2','y2','target_box','base_logits','base_margin',
               'predicted_class','alternative_class','categories']
missing_3c = [k for k in required_3c if k not in globals()]
if missing_3c:
    raise RuntimeError('Run Steps 3A and 3B first; missing: ' + ', '.join(missing_3c))

# HIGH-ITERATION preset: 500 spatial controls per region (original: 30),
# seven strength levels (original: three), up to five candidate regions.
# All of these increase evaluation thoroughness/computation, NOT model-training epochs.
# For a short test you can set N_CONTROLS_3C=4 in a separate Colab code cell.
N_CONTROLS_3C = int(globals().get('N_CONTROLS_3C', 500))
BATCH_3C = int(globals().get('BATCH_3C', 2))  # small batches protect Colab memory
STRENGTHS_3C = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0)
METHODS_3C = ('blur','neutral_fill')
MAX_REGIONS_3C = int(globals().get('MAX_REGIONS_3C', 5))
SEED_3C = 2026
BOOTSTRAP_REPETITIONS_3C = 1500  # quantile precision diagnostic, NOT confidence of causal effects
if N_CONTROLS_3C < 3 or BATCH_3C < 1:
    raise ValueError('Use N_CONTROLS_3C >= 3 and BATCH_3C >= 1.')

model.eval()
for p in model.parameters(): p.requires_grad_(False)
source_3c = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
H3, W3 = source_3c.shape[:2]
ch3, cw3 = int(y2-y1), int(x2-x1)
crop_3c = source_3c[y1:y2,x1:x2].copy()
fg_3c = np.asarray(foreground, dtype=bool).copy()
if fg_3c.shape != (ch3,cw3):
    raise RuntimeError('Foreground and object box sizes do not match; rerun Steps 3A/3B.')

# Broccoli-only green color gate helps remove nearby meat/container; NOT a segmentation ground truth.
mask_note_3c = 'GrabCut foreground estimate from Step 3B'
target_name_3c = str(categories[int(predicted_class)]).lower()
if target_name_3c == 'broccoli':
    hsv = cv2.cvtColor(crop_3c, cv2.COLOR_RGB2HSV)
    green_seed = cv2.inRange(hsv, (33,35,20), (100,255,255)) > 0
    # Proximity gate admits dark florets near green pixels, and is still a heuristic.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23))
    green_near = cv2.dilate(green_seed.astype(np.uint8), k) > 0
    proposed = fg_3c & green_near
    if proposed.sum() >= 0.12 * fg_3c.sum():
        fg_3c = proposed
        mask_note_3c = 'GrabCut intersected with green-proximity HEURISTIC (broccoli only)'
    else:
        mask_note_3c += ' (green gate rejected as too small)'
print('[Step 3C]', mask_note_3c)
print('[Step 3C] Foreground coverage of box:', round(float(fg_3c.mean()),3))

# Keep selected original SLIC regions, but only where the refined foreground agrees.
selected_3c = []
minimum_pixels = max(40, int(0.015 * fg_3c.sum()))
for rec in record:
    rid = int(rec['region'])
    m = (segments == rid) & fg_3c
    if m.sum() >= minimum_pixels:
        selected_3c.append((rid,m,int(m.sum())))
    if len(selected_3c) >= MAX_REGIONS_3C: break
if not selected_3c:
    raise RuntimeError('No foreground candidate remains after refinement; inspect the mask.')
print('[Step 3C] Testing:', [(r,n) for r,_,n in selected_3c])

# Prepare content replacements once; no generative reconstruction or hallucinated features.
blur_3c = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=14)).convert('RGB'), dtype=np.uint8)[y1:y2,x1:x2]
neutral_3c = np.median(crop_3c[fg_3c], axis=0).astype(np.uint8)
rng_3c = np.random.default_rng(SEED_3C)
fg_coords_3c = np.column_stack(np.where(fg_3c)).astype(np.int32)

# Controls: same number of pixels; compact nearest-foreground neighborhoods.
# IMPORTANT: same area != same shape / location / semantic evidence.
def control_mask_3c(n, original_mask):
    best = None
    best_overlap = 2.0
    for _ in range(30):
        yy,xx = fg_coords_3c[int(rng_3c.integers(len(fg_coords_3c)))].tolist()
        distances = ((fg_coords_3c[:,0].astype(np.float64)-yy)**2
                   + (fg_coords_3c[:,1].astype(np.float64)-xx)**2)
        idx = np.argpartition(distances, n-1)[:n]
        m = np.zeros((ch3,cw3), dtype=bool)
        m[fg_coords_3c[idx,0], fg_coords_3c[idx,1]] = True
        overlap = float((m & original_mask).sum()/n)
        if overlap < best_overlap:
            best,best_overlap=m,overlap
        if overlap <= 0.25: break
    return best,float(best_overlap)

# Inputs are generated ONE SMALL BATCH AT A TIME to avoid Colab crashes.
def edit_image_3c(mask, method, strength):
    edited = source_3c.copy()
    view = edited[y1:y2,x1:x2]
    old = crop_3c[mask].astype(np.float32)
    if method=='blur':
        target = blur_3c[mask].astype(np.float32)
    elif method=='neutral_fill':
        target = np.broadcast_to(neutral_3c, old.shape).astype(np.float32)
    else: raise ValueError(method)
    view[mask] = np.clip((1-strength)*old + strength*target,0,255).astype(np.uint8)
    return edited

@torch.inference_mode()
def fixed_roi_logits_batch_3c(images_np):
    ts = [to_tensor(PILImage.fromarray(a)).to(device) for a in images_np]
    transformed, _ = model.transform(ts, None)
    features = model.backbone(transformed.tensors)
    fixed_box = target_box.detach().to(device=device,dtype=torch.float32)
    roi_boxes = []
    for hh,ww in transformed.image_sizes:
        scale = fixed_box.new_tensor([ww/W3,hh/H3,ww/W3,hh/H3])
        roi_boxes.append((fixed_box*scale).unsqueeze(0))
    pooled = model.roi_heads.box_roi_pool(features,roi_boxes,transformed.image_sizes)
    embedding = model.roi_heads.box_head(pooled)
    logits, _ = model.roi_heads.box_predictor(embedding)
    return logits.detach().float().cpu().numpy()

# Detect any interface mismatch BEFORE paying for >100 inferences.
baseline_3c = fixed_roi_logits_batch_3c([source_3c])[0]
reference_3c = base_logits.detach().float().cpu().numpy()
max_disagreement_3c = float(np.max(np.abs(baseline_3c-reference_3c)))
print('[Step 3C] RoI logit consistency vs Step 3A:',round(max_disagreement_3c,6))
if max_disagreement_3c > 0.05:
    raise RuntimeError('RoI score mismatch with Step 3A; do NOT compare margins. Rerun Steps 1-3B.')
base_m_3c = float(baseline_3c[predicted_class]-baseline_3c[alternative_class])

jobs_3c=[]
control_overlap_3c={}
for rid,m,n in selected_3c:
    for mode in METHODS_3C:
        for s in STRENGTHS_3C:
            jobs_3c.append(dict(rid=rid,mask=m,mode=mode,strength=s,kind='observed',trial=-1))
    overlaps=[]
    for t in range(N_CONTROLS_3C):
        c,ov=control_mask_3c(n,m)
        overlaps.append(ov)
        for mode in METHODS_3C:
            jobs_3c.append(dict(rid=rid,mask=c,mode=mode,strength=1.0,kind='control',trial=t))
    control_overlap_3c[rid]=overlaps

# Resumable score cache. Exact job order is deterministic given the same data/config.
# Interrupted runs can be resumed in the SAME Colab file system / mounted Drive.
out_3c=Path(globals().get('C3R_OUTPUT_DIR', '/content/c3r_step1')); out_3c.mkdir(exist_ok=True,parents=True)
signature_3c = hashlib.sha256(
    source_3c.tobytes() + segments.astype(np.int32).tobytes() + fg_3c.tobytes()
    + reference_3c.astype(np.float32).tobytes()
    + json.dumps({'controls':N_CONTROLS_3C, 'strengths':STRENGTHS_3C,
                  'regions':[int(rid) for rid,_,_ in selected_3c],
                  'classes':[int(predicted_class), int(alternative_class)],
                  'seed':SEED_3C},sort_keys=True).encode('utf-8')
).hexdigest()[:16]
checkpoint_3c=out_3c/f'step3c_progress_{signature_3c}.jsonl'
finished_3c={}
if checkpoint_3c.exists():
    for line in checkpoint_3c.read_text(encoding='utf-8').splitlines():
        try:
            obj=json.loads(line)
            finished_3c[int(obj['job_id'])]=obj
        except (ValueError,KeyError,json.JSONDecodeError):
            pass
print('[Step 3C] Scoring',len(jobs_3c),'controlled image edits (batch=',BATCH_3C,');',
      len(finished_3c),'already cached.')
start_3c=time.time()
with checkpoint_3c.open('a', encoding='utf-8') as score_cache:
    for k in range(0,len(jobs_3c),BATCH_3C):
        portion=[(index,jobs_3c[index]) for index in range(k,min(k+BATCH_3C,len(jobs_3c)))
                 if index not in finished_3c]
        if portion:
            arrays=[edit_image_3c(j['mask'],j['mode'],j['strength']) for _,j in portion]
            logits_batch=fixed_roi_logits_batch_3c(arrays)
            for (index,j),logits in zip(portion,logits_batch):
                m_new=float(logits[predicted_class]-logits[alternative_class])
                result=dict(job_id=index,region=int(j['rid']),mode=j['mode'],strength=float(j['strength']),
                    kind=j['kind'],trial=int(j['trial']),margin=m_new,delta=base_m_3c-m_new)
                finished_3c[index]=result
                score_cache.write(json.dumps(result)+'\n')
            score_cache.flush()  # save completed batches for Colab disconnects
            del arrays,logits_batch
        if (k//BATCH_3C + 1)%25==0 or k+BATCH_3C>=len(jobs_3c):
            print(f' [Step 3C] {min(k+BATCH_3C,len(jobs_3c))}/{len(jobs_3c)} checked; {time.time()-start_3c:.0f}s')
        if device.type=='cuda' and (k//BATCH_3C)%60==0:
            torch.cuda.empty_cache()
records_3c=[finished_3c[i] for i in range(len(jobs_3c))]

# Conservative descriptive screening: two edits, seven strengths, 500 controls.
# The spatial controls may be correlated and are NOT independent statistical trials.
summary_3c=[]
bootstrap_rng_3c=np.random.default_rng(SEED_3C+93)
for rid,mask,n in selected_3c:
    bymode={}
    for mode in METHODS_3C:
        ob=sorted((r for r in records_3c if r['region']==rid and r['mode']==mode and r['kind']=='observed'),
                  key=lambda r:r['strength'])
        controls=np.array([r['delta'] for r in records_3c
            if r['region']==rid and r['mode']==mode and r['kind']=='control'],dtype=float)
        effect=float(ob[-1]['delta'])
        q95=float(np.quantile(np.abs(controls),.95))
        # Resampling the finite CONTROL POOL diagnoses how much its q95 estimate changes.
        # This is not a scientific CI for causality, robustness or a population effect.
        abs_controls=np.abs(controls)
        bq95=np.empty(BOOTSTRAP_REPETITIONS_3C,dtype=float)
        for bi in range(BOOTSTRAP_REPETITIONS_3C):
            sampled=bootstrap_rng_3c.choice(abs_controls,size=len(abs_controls),replace=True)
            bq95[bi]=np.quantile(sampled,.95)
        q95_range=[float(v) for v in np.quantile(bq95,[.025,.975])]
        stable_sign=bool(all(r['delta']*effect>0 for r in ob) and abs(effect)>0.05)
        exceed_fraction=float((1+np.count_nonzero(abs_controls>=abs(effect)))/(len(controls)+1))
        bymode[mode]=dict(full_effect=effect, doses=[dict(strength=r['strength'],delta=r['delta']) for r in ob],
            controls=controls.tolist(),q95_control_abs=q95,control_q95_bootstrap_interval=q95_range,
            descriptive_control_exceedance_fraction=exceed_fraction,
            outside_control_q95=bool(abs(effect)>q95),stable_sign=stable_sign)
    strong=(bymode['blur']['outside_control_q95'] and bymode['neutral_fill']['outside_control_q95']
            and bymode['blur']['stable_sign'] and bymode['neutral_fill']['stable_sign']
            and bymode['blur']['full_effect']*bymode['neutral_fill']['full_effect']>0
            and np.median(control_overlap_3c[rid]) <= 0.35)
    label=('provisional SUPPORT-LIKE region' if bymode['blur']['full_effect']>0
           else 'provisional CONFLICT-LIKE region') if strong else 'UNDETERMINED'
    if np.median(control_overlap_3c[rid]) > 0.35:
        print(' [Caution] Region',rid,': sampled controls substantially overlap the target region.')
    summary_3c.append(dict(region=rid,pixels=n,screening_label=label,
                           median_control_overlap=float(np.median(control_overlap_3c[rid])),methods=bymode))
    print(' Region',rid,'|',label,'| blur',round(bymode['blur']['full_effect'],3),
          '| fill',round(bymode['neutral_fill']['full_effect'],3),
          '| control q95',round(bymode['blur']['q95_control_abs'],3),round(bymode['neutral_fill']['q95_control_abs'],3))

# Six panel figure -- computed from real frozen detector scores, no invented numbers.
fig,axs=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
axs[0,0].imshow(source_3c)
axs[0,0].add_patch(Rectangle((x1,y1),cw3,ch3,fill=False,edgecolor='lime',lw=2))
axs[0,0].set_title('A. Clean / fixed target box')
axs[0,1].imshow(crop_3c)
axs[0,1].imshow(np.ma.masked_where(~fg_3c,fg_3c),cmap='Greens',alpha=.42)
axs[0,1].set_title('B. Refined foreground proxy (NOT ground truth)')
selected_colors=['red','deepskyblue','yellow','magenta','cyan']
axs[0,2].imshow(crop_3c)
for (rid,m,n),c in zip(selected_3c,selected_colors):
    axs[0,2].contour(m.astype(np.uint8),levels=[.5],colors=[c],linewidths=1.6)
    ys,xs=np.where(m)
    if len(xs): axs[0,2].text(float(np.median(xs)),float(np.median(ys)),f'R{rid}',color='white',fontsize=10,
                           bbox=dict(fc='black',alpha=.65,pad=1))
axs[0,2].set_title('C. Foreground-restricted candidates')
# real full-strength blur example for first region
rid0,m0,_=selected_3c[0]
axs[1,0].imshow(edit_image_3c(m0,'blur',1.0)[y1:y2,x1:x2])
axs[1,0].set_title(f'D. R{rid0}: actual blur intervention')
for mode,style in [('blur','-'),('neutral_fill','--')]:
    for s in summary_3c:
        vals=s['methods'][mode]['doses']
        axs[1,1].plot([v['strength'] for v in vals],[v['delta'] for v in vals],
                      marker='o',linestyle=style,label=f"R{s['region']} {mode}")
axs[1,1].axhline(0,color='black',lw=.8)
axs[1,1].set_xlabel('Intervention blend strength')
axs[1,1].set_ylabel('Δ logit margin (baseline − edited)')
axs[1,1].set_title('E. Dose-response on the same target RoI')
axs[1,1].legend(fontsize=7,ncol=2)
xs=[];ys=[];colors=[];names=[]
for s in summary_3c:
    for mode in METHODS_3C:
        names.append(f"R{s['region']}\n{mode.replace('neutral_fill','fill')}")
        xs.append(s['methods'][mode]['full_effect'])
        ys.append(s['methods'][mode]['q95_control_abs'])
        colors.append('tab:blue' if mode=='blur' else 'tab:orange')
idx=np.arange(len(xs))
axs[1,2].bar(idx,xs,color=colors,alpha=.85,label='Observed signed effect')
axs[1,2].scatter(idx,ys,marker='^',c='black',s=38,label='95th percentile |controls|')
axs[1,2].scatter(idx,-np.array(ys),marker='v',c='black',s=38)
axs[1,2].axhline(0,color='black',lw=.8)
axs[1,2].set_xticks(idx,names,fontsize=8)
axs[1,2].set_ylabel('Δ logit margin')
axs[1,2].set_title('F. Observed effects vs spatial controls')
axs[1,2].legend(fontsize=7)
for a in axs.flat:
    if a not in [axs[1,1],axs[1,2]]: a.axis('off')
fig.suptitle(f'C³R Step 3C | {categories[predicted_class]} vs {categories[alternative_class]} '
             f'| baseline logit margin {base_m_3c:+.3f}\n'
             'Controlled visual-region screening; semantic concepts, causality, and reliability NOT verified',fontsize=12)

fig_path_3c=out_3c/'step3c_region_screening.png'
json_path_3c=out_3c/'step3c_region_screening.json'
fig.savefig(fig_path_3c,dpi=180,bbox_inches='tight')
plt.show()
with open(json_path_3c,'w',encoding='utf-8') as f:
    json.dump(dict(image=str(globals().get('image_path','current image')),
        target_class=str(categories[predicted_class]),alternative_class=str(categories[alternative_class]),
        baseline_margin=base_m_3c,reference_consistency_error=max_disagreement_3c,
        foreground_mask_method=mask_note_3c,n_controls=N_CONTROLS_3C,
        regions=summary_3c,number_of_scored_edits=len(records_3c),strength_levels=STRENGTHS_3C,
        bootstrap_resamples=BOOTSTRAP_REPETITIONS_3C,semantic_concepts_verified=False,calibrated_reliability=None,
        cautions=['Color gate is a broccoli-specific heuristic, not segmentation ground truth',
                  '500 sampled spatial controls may be correlated, have repeated patterns and overlap target regions',
                  'Selection and testing share the same image; multiple candidate comparisons risk selection bias',
                  'Bootstrap interval measures finite-control quantile precision only; NOT statistical significance',
                  'Neutral fill and blur can be off-manifold',
                  'Object-specific semantic labels and cross-image validation remain pending']),f,indent=2)
print('\nSTEP 3C COMPLETED')
print('Figure:',fig_path_3c)
print('Data:',json_path_3c)
# Extra tidy CSV for statistical exploration and paper figures.
csv_path_3c=out_3c/'step3c_all_edits.csv'
with csv_path_3c.open('w',newline='',encoding='utf-8') as csv_file:
    writer=csv.DictWriter(csv_file,fieldnames=['job_id','region','mode','strength','kind','trial','margin','delta'])
    writer.writeheader()
    writer.writerows(records_3c)
print('Per-edit CSV:',csv_path_3c)
print('Resume checkpoint:',checkpoint_3c)
