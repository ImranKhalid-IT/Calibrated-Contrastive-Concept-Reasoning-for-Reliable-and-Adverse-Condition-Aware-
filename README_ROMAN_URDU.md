# C3R COCO128 — Separate Python Cells + High-Iteration Version

## Files aur run karne ka order

Notebook me original **5 code cells** hain. Neeche har cell ki alag `.py` file di gayi hai. **Step 3A ki original upload file notebook ke andar nahi thi**, is liye uska compatible, naya replacement alag banaya gaya hai.

| Order | File | Kaam |
| --- | --- | --- |
| 1 | `01_Detection.py` | COCO128 images download, pretrained frozen Faster R-CNN detection |
| 2 | `02_RoI_Features.py` | RoI features aur class-response maps |
| 3 | `03_Upload_Step3A.py` | Colab me `.py` upload karne ka button |
| Upload at Step 3 | **`03A_UPLOAD_THIS_Concept_Candidates.py`** | Jo Python file upload karni hai (NEW compatible Step 3A) |
| 4 | `04_Region_Validation.py` | GrabCut + zyada random-region controls |
| 5 | `05_High_Iteration_Screening.py` | 500 controls/region, 7 strengths, checkpoint, bootstrap diagnostics |

### Aasan tareeqa (recommended)

1. `C3R_High_Iteration_Colab.ipynb` ko Google Colab me open karein.
2. `Runtime > Change runtime type > T4 GPU` (agar available ho) select karein. CPU par bhi chalega, lekin heavy experiments bahut slow honge.
3. Cells ko **order me 1, 2, 3, 4, 5** run karein.
4. **Cell 3 par file upload dialog aayega**: yahan `03A_UPLOAD_THIS_Concept_Candidates.py` select karein. File automatically same runtime me execute hogi.
5. Step 3A complete hone ke baad Step 3B (cell 4), aur phir Step 3C (cell 5) run karein.
6. Outputs default folder `/content/c3r_step1/` me save honge: PNG figures, JSON results, 3C CSV, aur resumable JSONL checkpoints.

### Agar `.py` scripts individual run karni hain

Colab me pehle ye `.py` files upload karein aur **same runtime / same notebook namespace** me run karein. Colab code cell me `%run -i /content/01_Detection.py` phir `%run -i /content/02_RoI_Features.py`. Third file `03_Upload_Step3A.py` ko `%run -i` se run karna hai (upload prompt dikhega), phir `%run -i /content/04_Region_Validation.py` aur `%run -i /content/05_High_Iteration_Screening.py`. Different Python processes me plain `python file.py` chalane se previous-cell variables nahi milenge.

## Iterations / controls (original se barhaye gaye)

| Parameter | Pehle | Naya default |
| --- | ---: | ---: |
| Step 3B random controls per region | 3 | **120** |
| Step 3C random controls per region | 30 | **500** |
| Step 3C intervention strengths | 3 | **7** |
| Step 3C max candidate regions | 3 | **5** |
| Step 3C bootstrap quantile resamples | None | **1,500** |
| Step 3A requested SLIC regions | External code missing | **48** |

Step 3B me 5 qualify hone wale regions ke liye taqriban **1,210** image-edit evaluations, aur Step 3C me **5,070** image-edit evaluations ho sakte hain (asal count selected regions par depend karta hai). Existing original notebook ki Step 3C example me 198 edits the. Heavy experiment **GPU hours / Colab resources** le sakta hai; notebook me koi result guarantee nahi kiya gaya.

### Kam ya aur zyada workload kaise set karen

**Step 3B se pehle** Colab me ek code cell chala sakte hain:

```python
N_CONTROLS_3B = 120    # Quick check: 4 ; heavier: 250
MAX_REGIONS_3B = 5
```

**Step 3C se pehle** Colab me ek code cell chala sakte hain:

```python
N_CONTROLS_3C = 500    # Quick check: 4 ; more exhaustive: 1000
MAX_REGIONS_3C = 5
BATCH_3C = 2           # GPU memory bachane ke liye chhota rakhein
```

### Long Colab run ka progress Google Drive me save karna

Colab ka `/content/` **runtime disconnect ke baad reset ho sakta hai**. Checkpoints ko zyada safely bachane ke liye **Step 1 se pehle** ek naya code cell run karein:

```python
from google.colab import drive
drive.mount('/content/drive')
C3R_OUTPUT_DIR = '/content/drive/MyDrive/C3R_Long_Runs'
```

Is se dataset aur results Drive folder me honge. Same image, script, settings aur model baseline ke saath dobara run karne par Step 3B/3C same fingerprint wala progress JSONL checkpoint read karte hain aur completed edits skip karte hain. Model ya config change hone par different checkpoint ban sakta hai.

## Results aur scientific caveats

- Faster R-CNN **pretrained aur frozen** hai. Is pipeline me `epochs` ka training concept nahi; controls/iterations barhane ka matlab aur zyada *evaluation* hai, **training accuracy guarantee nahi**.
- Step 3C me **dose-response**, spatial controls ke **95th percentile**, basic descriptive exceedance fractions, aur bootstrap *control-quantile precision* report hoti hai. Ye formal p-values, causal identification, semantic concept verification ya calibrated reliability nahi.
- COCO128 ki ek image par region selection aur usi image par testing hone ki wajah se selection bias possible hai. Scientific paper ke liye independent held-out images, proper segmentation validation, ablations, multiple seeds aur uncertainty calibration alag chahiye.
- Step 3A file original upload wale source ke bina **fresh compatible implementation** hai. Is se original paper/notebook ke exact Step 3A results reproduce hona zaroori nahi.
- `requirements.txt` environment details ke liye hai; Colab me PyTorch aksar installed hota hai. Bina zaroorat torch ko reinstall na karein.
