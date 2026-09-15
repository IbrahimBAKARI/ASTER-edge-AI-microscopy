# ===== execution [1] =====
import os, sys, json, time, random, hashlib, subprocess
from pathlib import Path

SEED = 42
random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)

import torch, numpy as np
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
print("torch:", torch.__version__)
# ===== execution [2] =====
# --- bridge credentials, then bootstrap the repository ---------------------------------
# Paste the two values printed by `bash data_bridge/serve.sh` on the Mac.
BRIDGE_URL   = "https://<redacted>.trycloudflare.com"
BRIDGE_TOKEN = "<redacted>"

# Chicken and egg: the repo's own downloader lives inside the repo, and the repo arrives
# over the bridge. This bootstrap fetches the `payload` dataset - the repository zip and
# the block-1 weights - with the standard library alone; everything after it uses the
# repo's tooling.
import io, tarfile, urllib.request, zipfile

def _bridge(path):
    request = urllib.request.Request(f"{BRIDGE_URL.rstrip('/')}/{path}",
                                     headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"})
    return urllib.request.urlopen(request, timeout=300)

PAYLOAD = DATA/"payload"
if not (PAYLOAD/"aster-block2.zip").exists():
    assert BRIDGE_URL and BRIDGE_TOKEN, "paste BRIDGE_URL and BRIDGE_TOKEN above"
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    plan = json.load(_bridge("shards/payload.json"))
    for shard in plan["shards"]:
        blob = _bridge(f"tar/payload/{shard['id']}").read()
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            archive.extractall(PAYLOAD)
    print("payload:", sorted(p.name for p in PAYLOAD.iterdir()))
if not (REPO/"data_bridge/colab_pull.py").exists():
    with zipfile.ZipFile(PAYLOAD/"aster-block2.zip") as archive:
        archive.extractall(REPO)
assert (REPO/"data_bridge/colab_pull.py").exists(), "the repository zip is incomplete"

WEIGHTS = PAYLOAD/"wbc_detector.pt"          # used by section 2 for the x40 crops
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"datasets"))

# Install only what the Colab image lacks. `|| true`: a pip failure must never abort an
# unattended run, and the image ships its own torch/numpy - pinning against it makes pip
# refuse the whole set.
!pip -q install -r {REPO}/env/requirements-colab.txt || true
import importlib
for _module in ("sklearn", "scipy", "pandas", "yaml", "onnxruntime", "ultralytics"):
    try:
        importlib.import_module(_module)
    except ImportError:
        print(f"  MISSING: {_module}")

from aster_block2.grid import Thresholds, evaluate, load_thresholds
from aster_block2.proportion_test import test_proportion, min_count_to_assert
from aster_block2.models import Encoder, CellHead, GatedAttentionMIL, partial_label_loss, attention_entropy_penalty, encode_bag
from aster_block2.preprocess import build_eval_transform, IMAGENET_MEAN, IMAGENET_STD
from aster_block2.schema import SessionResult
print("aster_block2 imported from", REPO)
# ===== execution [3] =====
from google.colab import drive
drive.mount("/content/drive")

# Everything durable lives in Drive so a disconnect never costs a training run.
WORK  = Path("/content/drive/MyDrive/aster_block2")      # checkpoints, results, logs
DATA  = Path("/content/data")                             # corpus, local SSD, fast
REPO  = Path("/content/aster-block2")                     # this repository
for path in (WORK, DATA, WORK/"checkpoints", WORK/"results", WORK/"logs"):
    path.mkdir(parents=True, exist_ok=True)
print(WORK, DATA, REPO)
# ===== execution [4] =====
# --- bridge credentials, then bootstrap the repository ---------------------------------
# Paste the two values printed by `bash data_bridge/serve.sh` on the Mac.
BRIDGE_URL   = "https://<redacted>.trycloudflare.com"
BRIDGE_TOKEN = "<redacted>"

# Chicken and egg: the repo's own downloader lives inside the repo, and the repo arrives
# over the bridge. This bootstrap fetches the `payload` dataset - the repository zip and
# the block-1 weights - with the standard library alone; everything after it uses the
# repo's tooling.
import io, tarfile, urllib.request, zipfile

def _bridge(path):
    request = urllib.request.Request(f"{BRIDGE_URL.rstrip('/')}/{path}",
                                     headers={"Authorization": f"Bearer {BRIDGE_TOKEN}"})
    return urllib.request.urlopen(request, timeout=300)

PAYLOAD = DATA/"payload"
if not (PAYLOAD/"aster-block2.zip").exists():
    assert BRIDGE_URL and BRIDGE_TOKEN, "paste BRIDGE_URL and BRIDGE_TOKEN above"
    PAYLOAD.mkdir(parents=True, exist_ok=True)
    plan = json.load(_bridge("shards/payload.json"))
    for shard in plan["shards"]:
        blob = _bridge(f"tar/payload/{shard['id']}").read()
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            archive.extractall(PAYLOAD)
    print("payload:", sorted(p.name for p in PAYLOAD.iterdir()))
if not (REPO/"data_bridge/colab_pull.py").exists():
    with zipfile.ZipFile(PAYLOAD/"aster-block2.zip") as archive:
        archive.extractall(REPO)
assert (REPO/"data_bridge/colab_pull.py").exists(), "the repository zip is incomplete"

WEIGHTS = PAYLOAD/"wbc_detector.pt"          # used by section 2 for the x40 crops
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"datasets"))

# Install only what the Colab image lacks. `|| true`: a pip failure must never abort an
# unattended run, and the image ships its own torch/numpy - pinning against it makes pip
# refuse the whole set.
!pip -q install -r {REPO}/env/requirements-colab.txt || true
import importlib
for _module in ("sklearn", "scipy", "pandas", "yaml", "onnxruntime", "ultralytics"):
    try:
        importlib.import_module(_module)
    except ImportError:
        print(f"  MISSING: {_module}")

from aster_block2.grid import Thresholds, evaluate, load_thresholds
from aster_block2.proportion_test import test_proportion, min_count_to_assert
from aster_block2.models import Encoder, CellHead, GatedAttentionMIL, partial_label_loss, attention_entropy_penalty, encode_bag
from aster_block2.preprocess import build_eval_transform, IMAGENET_MEAN, IMAGENET_STD
from aster_block2.schema import SessionResult
print("aster_block2 imported from", REPO)
# ===== execution [5] =====
# --- the run directory: everything this run leaves behind, in Drive, as it happens -----
from aster_block2.runlog import RunDir

run = RunDir.create(WORK, repo=REPO)
run.install_exception_hook()   # any cell error lands in ALERTS.md, even unattended

# The run writes ~0.4 GB to Drive (weights, ONNX, logs, results). A full Drive makes the
# FIRST checkpoint fail hours into training - better to know now.
import shutil as _shutil
_drive_free = _shutil.disk_usage("/content/drive/MyDrive").free / 1e9
print(f"Drive: {_drive_free:.2f} GB free")
assert _drive_free > 0.6, (f"only {_drive_free:.2f} GB free in Drive; the run needs ~0.4 GB "
                           f"for weights and results. Free some space and re-run.")
print(f"""
Tout est ecrit dans {run.root} au fil de l'eau :
  logs/        stdout de chaque section + traceback si ca casse
  metrics/     une ligne JSON par epoque -> les courbes survivent a une deconnexion
  checkpoints/ par epoque, les 2 dernieres gardees + <nom>_final.pt
  results/     chaque CSV/JSON que l'article cite
  bundle/      exactement ce que integration/PATCH.md attend sur le Jetson
  manifest.json + env/  versions, GPU, seeds, empreintes de la pre-inscription
Une deconnexion ne coute donc jamais plus que l'epoque en cours.
""")
# ===== execution [6] =====
# The frozen grid must be the frozen grid. This aborts if anything drifted.
expected = {}
for line in (REPO/"results/PREREGISTRATION.sha256").read_text().splitlines():
    if line.startswith("#") or not line.strip(): continue
    digest, name = line.split(None, 1)
    expected[name.strip()] = digest

for name, digest in expected.items():
    actual = hashlib.sha256((REPO/name).read_bytes()).hexdigest()
    status = "OK" if actual == digest else "CHANGED"
    print(f"  {status:8} {name}")
    assert actual == digest, f"{name} differs from the frozen pre-registration"
GRID_SHA = expected["src/aster_block2/decision_grid.yaml"]
print("\ngrid sha256:", GRID_SHA)
# ===== execution [7] =====
# --- 1. the corpus and the x40 fields ---------------------------------------------------
# In-process, not subprocess: Colab's sys.executable is a different interpreter from the
# kernel and a child's stderr is swallowed. No existence guard: manifest.csv arrives in ONE
# shard, so its presence does not mean the corpus is complete; pull_dataset is itself
# idempotent and resumes shard by shard.
sys.path.insert(0, str(REPO/"data_bridge"))
import colab_pull

for _name in ("corpus", "x40_fields"):
    colab_pull.pull_dataset(BRIDGE_URL, BRIDGE_TOKEN, _name, DATA, workers=4)
# ===== execution [8] =====
# --- the manifest, merged with the splits ---------------------------------------------
import pandas as pd
manifest = pd.read_csv(DATA/"corpus/manifest.csv")
splits   = pd.read_csv(DATA/"corpus/splits.csv")
manifest = manifest.merge(splits[["corpus_path","split"]], on="corpus_path", how="left")
print(manifest.groupby(["source","split"]).size().to_string())
print("\ntotal images:", len(manifest))
# ===== execution [9] =====
# Completion is a MARKER, not the existence of the output directory: the directory is
# created before the loop, so a crash mid-way used to leave a partial crop set that the
# next run reported as "already present" - an incomplete stress test, silently.
import shutil
X40_OUT = DATA/"x40_sessions"
DONE = X40_OUT/".complete"
if not DONE.exists():
    shutil.rmtree(X40_OUT, ignore_errors=True)        # discard any partial result
    !pip -q install ultralytics
    from ultralytics import YOLO
    from PIL import Image
    from aster_block2.crops import crops_from_boxes   # the deployed crop convention

    assert WEIGHTS.exists(), "wbc_detector.pt missing from the bridge payload"
    detector = YOLO(str(WEIGHTS), task="detect")

    fields = sorted(p for p in (DATA/"x40_fields").rglob("*")
                    if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    print(len(fields), "x40 fields")

    CONF, IOU, IMGSZ = 0.18, 0.50, 960        # config/inference.yaml
    crop_dir = X40_OUT/"x40_all"/"crops"; crop_dir.mkdir(parents=True, exist_ok=True)
    kept, unreadable = 0, []
    for index, field in enumerate(fields):
        # One source field (train/slide_20260908_162808_658.jpg) has a broken JPEG stream.
        # OpenCV - what YOLO reads with - tolerates it; Pillow - what the crop is cut with -
        # does not. An unreadable field is excluded and named, never half-decoded.
        try:
            with Image.open(field) as _image:
                _image.convert("RGB")
        except OSError as exc:
            unreadable.append(f"{field.relative_to(DATA/'x40_fields')}: {exc}")
            continue
        boxes = [b.xyxy[0].tolist() for b in detector.predict(
            str(field), imgsz=IMGSZ, conf=CONF, iou=IOU, classes=[0], verbose=False)[0].boxes]
        # crops_from_boxes applies expand_and_clip_box + extract_crop exactly as the
        # deployed pipeline does: +10%/side, clipped, native pixels, and a degenerate box
        # drops the detection entirely rather than yielding a padded stub.
        for _, crop in crops_from_boxes(field, boxes):
            crop.save(crop_dir/f"field{index:03d}_wbc_{kept:04d}.png")
            kept += 1
    if unreadable:
        run.alert(f"{len(unreadable)} x40 field(s) unreadable, excluded from the stress test",
                  detail="\n".join(unreadable))
    usable = len(fields) - len(unreadable)
    DONE.write_text(f"{kept} crops from {usable} fields, {len(unreadable)} unreadable\n")
    print(f"x40 crops: {kept} from {usable} fields ({kept/max(usable,1):.2f} WBC/field), "
          f"{len(unreadable)} unreadable")
else:
    print("x40 crops already complete:", DONE.read_text().strip())
# ===== execution [10] =====
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

CELL_CLASSES = ["myeloblast","lymphoblast","promyelocyte","promyelocyte_abnormal",
                "myelocyte","metamyelocyte","band_neutrophil","segmented_neutrophil",
                "basophil","eosinophil","monocyte","lymphocyte","lymphocyte_atypical",
                "smudge_cell","erythroblast","other_artifact"]
CLASS_INDEX = {name: i for i, name in enumerate(CELL_CLASSES)}

train_tf = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(20)], p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.25),  # stain variation
    T.RandomApply([T.GaussianBlur(5, (0.1, 2.5))], p=0.4),                  # magnification proxy
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])
eval_tf = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class CellDataset(Dataset):
    """Cells with a definite OR partial label. `allowed` is a 0/1 mask over classes."""
    def __init__(self, frame, root, transform):
        self.rows = frame.reset_index(drop=True); self.root = Path(root); self.tf = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = Image.open(self.root/row.corpus_path).convert("RGB")
        allowed = torch.zeros(len(CELL_CLASSES))
        for name in str(row.cell_labels).split("|"):
            if name in CLASS_INDEX: allowed[CLASS_INDEX[name]] = 1.0
        return self.tf(image), allowed, torch.tensor(float(row.weight))

cell_rows = manifest[manifest.cell_labels.notna() & (manifest.cell_labels != "")]
train_rows = cell_rows[cell_rows.split == "cell_train"]
val_rows   = cell_rows[cell_rows.split == "cell_val"]
print(f"cell head: {len(train_rows)} train / {len(val_rows)} val")
print(train_rows.cell_labels.value_counts().head(20).to_string())
# ===== execution [11] =====
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score
from aster_block2.quantify import quantification_error

CKPT = WORK/"checkpoints/cell_head.pt"
EPOCHS_CELL, PATIENCE_CELL = 60, 8

# Monitored metric: MACRO F1 over the definite-label validation cells. Not macro RECALL,
# which was the first choice and was wrong: it rewards OVER-predicting rare classes, and
# over-predicting smudge cells is exactly what makes R3 fire on normal blood. The grid
# consumes PROPORTIONS, so precision matters as much as recall. The quantification MAE is
# logged alongside as the diagnostic closest to what the grid actually reads.
# (kept for the record) macro recall over the definite-label validation cells - the unweighted
# mean over classes. Overall accuracy would be dominated by segmented_neutrophil (15 253
# cells) and would happily ignore smudge_cell, lymphocyte_atypical and
# promyelocyte_abnormal - precisely the classes the chronic rules and the APL flag depend
# on. Macro recall makes the rare classes count as much as the common ones.

def make_balanced_sampler(frame):
    counts = frame.cell_labels.value_counts()
    weights = frame.cell_labels.map(lambda k: 1.0/counts[k]).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(frame), replacement=True)

encoder = Encoder(pretrained=True).to(DEVICE)
cell_head = CellHead().to(DEVICE)

if CKPT.exists():
    state = torch.load(CKPT, map_location=DEVICE)
    encoder.load_state_dict(state["encoder"]); cell_head.load_state_dict(state["cell_head"])
    print("loaded", CKPT)
else:
    train_loader = DataLoader(CellDataset(train_rows, DATA/"corpus", train_tf), batch_size=64,
                              sampler=make_balanced_sampler(train_rows), num_workers=2, drop_last=True)
    val_loader   = DataLoader(CellDataset(val_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(cell_head.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_CELL)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE=="cuda")

    best, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, EPOCHS_CELL+1):
        encoder.train(); cell_head.train(); running = 0.0
        for images, allowed, weights in train_loader:
            images, allowed, weights = images.to(DEVICE), allowed.to(DEVICE), weights.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=DEVICE=="cuda"):
                logits = cell_head(encoder(images))["logits"]
                log_probabilities = torch.log_softmax(logits, -1)
                mass = torch.logsumexp(log_probabilities.masked_fill(allowed==0, -1e30), -1)
                loss = -(mass * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        scheduler.step()

        encoder.eval(); cell_head.eval()
        preds, truth, in_allowed, total = [], [], 0, 0
        with torch.inference_mode():
            for images, allowed, _ in val_loader:
                logits = cell_head(encoder(images.to(DEVICE)))["logits"].cpu()
                p = logits.argmax(-1)
                in_allowed += allowed[torch.arange(len(p)), p].sum().item(); total += len(p)
                definite = allowed.sum(1) == 1
                if definite.any():
                    preds += p[definite].tolist(); truth += allowed[definite].argmax(-1).tolist()
        macro = f1_score(truth, preds, average="macro", zero_division=0) if truth else 0.0
        macro_recall = recall_score(truth, preds, average="macro", zero_division=0) if truth else 0.0
        from collections import Counter as _C
        qmae = quantification_error(_C(truth), _C(preds)) if truth else 1.0
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.4f}  "
              f"macro-F1 {macro:.4f}  macro-recall {macro_recall:.4f}  quant-MAE {qmae:.4f}"
              + ("   <- best" if macro > best else f"   (patience {waited+1}/{PATIENCE_CELL})"))
        run.metric("cell_head", epoch=epoch, loss=running/len(train_loader),
                   val_macro_f1=macro, val_macro_recall=macro_recall,
                   val_quantification_mae=qmae, val_in_allowed_set=in_allowed/total,
                   lr=scheduler.get_last_lr()[0])

        if macro > best:
            best, best_epoch, waited = macro, epoch, 0
            best_state = {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                          "cell_head": {k: v.detach().cpu().clone() for k, v in cell_head.state_dict().items()}}
            run.checkpoint("cell_head", best_state, epoch=epoch, keep_last=1)
        else:
            waited += 1
            if waited >= PATIENCE_CELL:
                print(f"early stop at epoch {epoch}; best was epoch {best_epoch} "
                      f"(macro-recall {best:.4f})")
                break

    encoder.load_state_dict(best_state["encoder"]); cell_head.load_state_dict(best_state["cell_head"])
    encoder.to(DEVICE); cell_head.to(DEVICE)
    torch.save(best_state, CKPT)
    run.result("cell_head_training.json",
               {"best_epoch": best_epoch, "best_macro_f1": best, "epochs_run": epoch,
                "max_epochs": EPOCHS_CELL, "patience": PATIENCE_CELL,
                "monitor": "macro F1 on definite-label cell_val"})
    print("saved", CKPT, f"(best epoch {best_epoch})")
# ===== execution [12] =====
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import spearmanr

# --- per-class report on definite-label validation cells ---------------------
definite = val_rows[~val_rows.cell_labels.str.contains(r"\|", na=False)]
loader = DataLoader(CellDataset(definite, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
predictions, truths = [], []
encoder.eval(); cell_head.eval()
with torch.inference_mode():
    for images, allowed, _ in loader:
        predictions += cell_head(encoder(images.to(DEVICE)))["logits"].argmax(-1).cpu().tolist()
        truths += allowed.argmax(-1).tolist()
report = classification_report(truths, predictions, labels=list(range(len(CELL_CLASSES))),
                               target_names=CELL_CLASSES, output_dict=True, zero_division=0)
import pandas as pd
report_frame = pd.DataFrame(report).T
print(report_frame.to_string())
report_frame.to_csv(WORK/"results/cell_head_per_class.csv")
run.result("cell_head_per_class.csv", report_frame.reset_index())
# ===== execution [13] =====
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import spearmanr

# --- per-class report on definite-label validation cells ---------------------
definite = val_rows[~val_rows.cell_labels.str.contains(r"\|", na=False)]
loader = DataLoader(CellDataset(definite, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
predictions, truths = [], []
encoder.eval(); cell_head.eval()
with torch.inference_mode():
    for images, allowed, _ in loader:
        predictions += cell_head(encoder(images.to(DEVICE)))["logits"].argmax(-1).cpu().tolist()
        truths += allowed.argmax(-1).tolist()
report = classification_report(truths, predictions, labels=list(range(len(CELL_CLASSES))),
                               target_names=CELL_CLASSES, output_dict=True, zero_division=0)
import pandas as pd
report_frame = pd.DataFrame(report).T
print(report_frame.to_string())
report_frame.to_csv(WORK/"results/cell_head_per_class.csv")
run.result("cell_head_per_class.csv", report_frame.reset_index())
# ----- Out[13] -----
# PosixPath('/content/drive/MyDrive/aster_block2/runs/20260911_051357Z/results/cell_head_per_class.csv')
# ===== execution [14] =====
# --- the quantifier: what turns cell predictions into defensible PROPORTIONS ----------
# Fitted on the same definite-label validation cells, per GRID QUANTITY - blast_frac is a
# sum of three classes, so its error rate must be measured on that sum. This is what stops
# R3 from firing on a normal smear because 5 % of leukocytes were miscalled smudge cells.
from aster_block2.quantify import Quantifier
from aster_block2.grid import QUANTITIES

true_names = [CELL_CLASSES[i] for i in truths]
pred_names = [CELL_CLASSES[i] for i in predictions]
quantifier = Quantifier.fit_groups(true_names, pred_names, QUANTITIES)
quantifier.save(WORK/"checkpoints/quantifier.json")
run.result("quantifier.json", quantifier.report())

print(f"{'quantity':<18}{'TPR':>8}{'FPR':>8}{'sep':>8}   quantifiable")
for name, r in quantifier.report().items():
    flag = "yes" if r["quantifiable"] else "NO  <- the grid must abstain on this criterion"
    print(f"{name:<18}{r['tpr']:>8.3f}{r['fpr']:>8.3f}{r['separation']:>8.3f}   {flag}")
unquantifiable = [k for k, v in quantifier.report().items() if not v["quantifiable"]]
if unquantifiable:
    run.alert(f"{unquantifiable} cannot support a proportion claim (TPR - FPR < 0.05). "
              f"Any rule resting on them returns indeterminate by construction - report "
              f"it, do not retune.")
# ===== execution [15] =====
# --- bag differential vs the AML-MLL manual differential ---------------------
PB_MAP = {  # manual differential column -> our classes
    "pb_myeloblast": ["myeloblast"], "pb_promyelocyte": ["promyelocyte","promyelocyte_abnormal"],
    "pb_myelocyte": ["myelocyte"], "pb_metamyelocyte": ["metamyelocyte"],
    "pb_neutrophil_band": ["band_neutrophil"], "pb_neutrophil_segmented": ["segmented_neutrophil"],
    "pb_eosinophil": ["eosinophil"], "pb_basophil": ["basophil"], "pb_monocyte": ["monocyte"],
    "pb_lymph_typ": ["lymphocyte"], "pb_lymph_atyp_react": ["lymphocyte_atypical"],
}

def predict_counts(paths, batch=128):
    counts = {name: 0 for name in CELL_CLASSES}
    for start in range(0, len(paths), batch):
        images = torch.stack([eval_tf(Image.open(DATA/"corpus"/p).convert("RGB"))
                              for p in paths[start:start+batch]]).to(DEVICE)
        with torch.inference_mode():
            for index in cell_head(encoder(images))["logits"].argmax(-1).cpu().tolist():
                counts[CELL_CLASSES[index]] += 1
    return counts

mll = manifest[manifest.source == "aml_mll"].copy()
mll["differential"] = mll.extra.map(lambda s: json.loads(s).get("differential", {}))
rows = []
for patient, group in mll.groupby("patient_id"):
    manual = group.differential.iloc[0]
    if not manual or not manual.get("pb_total"): continue
    counts = predict_counts(group.corpus_path.tolist())
    total = sum(counts[c] for c in CELL_CLASSES)
    if not total: continue
    record = {"patient_id": patient, "n_cells": total}
    for column, classes in PB_MAP.items():
        record[f"manual_{column}"] = float(manual.get(column) or 0)
        record[f"pred_{column}"] = 100.0 * sum(counts[c] for c in classes) / total
    rows.append(record)
differential_frame = pd.DataFrame(rows)
differential_frame.to_csv(WORK/"results/differential_validation.csv", index=False)

print(f"{'column':<28}{'rho':>8}{'p':>10}   n={len(differential_frame)}")
correlations = {}
for column in PB_MAP:
    rho, pvalue = spearmanr(differential_frame[f"manual_{column}"], differential_frame[f"pred_{column}"])
    correlations[column] = rho
    print(f"{column:<28}{rho:>8.3f}{pvalue:>10.2g}")
json.dump(correlations, open(WORK/"results/differential_spearman.json","w"), indent=2)
run.result("differential_spearman.json", correlations)
run.result("differential_validation.csv", differential_frame)

if correlations["pb_myeloblast"] < 0.5:
    run.alert(f"criterion 6.5 MET: rho(myeloblast) = {correlations['pb_myeloblast']:.3f} < 0.5. "
              f"The grid is withdrawn; report P_abn only. Do not retune.",
              severity="FALSIFIER")
# ===== execution [16] =====
# 166 MB and recomputable in minutes from the encoder: local disk, not Drive.
FEATURES = DATA/"features_mll.pt"
if FEATURES.exists():
    cache = torch.load(FEATURES)
else:
    cache = {}
    encoder.eval()
    for patient, group in mll.groupby("patient_id"):
        paths = group.corpus_path.tolist(); embeddings = []
        for start in range(0, len(paths), 128):
            images = torch.stack([eval_tf(Image.open(DATA/"corpus"/p).convert("RGB"))
                                  for p in paths[start:start+128]]).to(DEVICE)
            with torch.inference_mode():
                embeddings.append(encoder(images).cpu())
        cache[patient] = torch.cat(embeddings)
    torch.save(cache, FEATURES)
print(len(cache), "patients cached,", sum(v.shape[0] for v in cache.values()), "cells")

patient_split = dict(zip(splits[splits.source=="aml_mll"].patient_id, splits[splits.source=="aml_mll"].split))
patient_label = {p: g.bag_label.iloc[0] for p, g in mll.groupby("patient_id")}
is_positive  = {p: 0 if label == "control" else 1 for p, label in patient_label.items()}
train_patients = [p for p in cache if patient_split.get(p) == "train_mil"]
fit_patients   = [p for p in cache if patient_split.get(p) == "dev_fit"]
cal_patients   = [p for p in cache if patient_split.get(p) == "dev_cal"]
print(f"MIL: {len(train_patients)} train / {len(fit_patients)} dev_fit / {len(cal_patients)} dev_cal")
# ===== execution [17] =====
from sklearn.metrics import roc_auc_score

MIL_CKPT = WORK/"checkpoints/mil_head.pt"
BAG, EPOCHS_MIL, PATIENCE_MIL = 200, 300, 30

# Early stopping is model selection, so it is paid for out of the TRAINING budget:
# mil_val patients, carved from train_mil by build_splits.py. Stopping on dev_fit would
# choose the epoch on the same patients that later fix tau_abn and theta_APL, and those
# operating points would look better than they are.
val_patients = [p for p in cache if patient_split.get(p) == "mil_val"]
print(f"early stopping on {len(val_patients)} mil_val patients "
      f"({sum(is_positive[p] for p in val_patients)} positive)")

mil = GatedAttentionMIL().to(DEVICE)
if MIL_CKPT.exists():
    mil.load_state_dict(torch.load(MIL_CKPT, map_location=DEVICE)); print("loaded", MIL_CKPT)
else:
    optimizer = torch.optim.AdamW(mil.parameters(), lr=1e-4, weight_decay=1e-4)
    generator = random.Random(SEED)
    positives = [p for p in train_patients if is_positive[p]]
    negatives = [p for p in train_patients if not is_positive[p]]

    def evaluate_on(patients):
        mil.eval(); scores, labels = [], []
        with torch.inference_mode():
            for patient in patients:
                scores.append(mil(cache[patient][:BAG].to(DEVICE))["probability"].item())
                labels.append(is_positive[patient])
        return roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")

    best, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, EPOCHS_MIL+1):
        mil.train(); losses = []
        order = [p for pair in zip(generator.sample(positives, len(positives)),
                                   generator.choices(negatives, k=len(positives))) for p in pair]
        for patient in order:
            features = cache[patient]
            index = torch.randperm(len(features), generator=torch.Generator().manual_seed(
                generator.randrange(10**9)))[:BAG]
            bag = features[index].to(DEVICE)
            output = mil(bag)
            target = torch.tensor(float(is_positive[patient]), device=DEVICE)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logit"], target)
            loss = loss + 0.1 * attention_entropy_penalty(output["entropy"], len(bag))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(loss.item())

        auroc = evaluate_on(val_patients)
        run.metric("mil_head", epoch=epoch, loss=float(np.mean(losses)), mil_val_auroc=auroc)
        if epoch % 5 == 0 or auroc > best:
            print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  mil_val AUROC {auroc:.4f}"
                  + ("   <- best" if auroc > best else f"   (patience {waited+1}/{PATIENCE_MIL})"))
        if auroc > best:
            best, best_epoch, waited = auroc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()}
            run.checkpoint("mil_head", best_state, epoch=epoch, keep_last=1)
        else:
            waited += 1
            if waited >= PATIENCE_MIL:
                print(f"early stop at epoch {epoch}; best was epoch {best_epoch} (AUROC {best:.4f})")
                break

    mil.load_state_dict(best_state); mil.to(DEVICE)
    torch.save(best_state, MIL_CKPT)
    run.result("mil_head_training.json",
               {"best_epoch": best_epoch, "best_mil_val_auroc": best, "epochs_run": epoch,
                "max_epochs": EPOCHS_MIL, "patience": PATIENCE_MIL,
                "monitor": "AUROC on mil_val patients (never dev_fit)",
                "n_val_patients": len(val_patients)})
    print(f"saved {MIL_CKPT} (best epoch {best_epoch}, mil_val AUROC {best:.4f})")
# ===== execution [18] =====
import yaml
from sklearn.metrics import roc_curve

thresholds = load_thresholds(REPO/"src/aster_block2/decision_grid.yaml")
resolved = {}

# --- tau_abn: specificity >= 0.95 on dev_fit ---------------------------------
mil.eval(); scores, labels = [], []
with torch.inference_mode():
    for patient in fit_patients:
        scores.append(mil(cache[patient][:BAG].to(DEVICE))["probability"].item())
        labels.append(is_positive[patient])
fpr, tpr, cuts = roc_curve(labels, scores)
resolved["p_abn"] = float(cuts[np.where(fpr <= 0.05)[0][-1]])
print(f"tau_abn = {resolved['p_abn']:.4f}  (specificity >= 0.95 on {len(fit_patients)} dev_fit patients)")

# --- theta_APL: specificity >= 0.95 among AML on dev_fit ---------------------
apl_fraction, apl_label = [], []
for patient in fit_patients:
    if patient_label[patient] == "control": continue
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    total = sum(counts[c] for c in CELL_CLASSES) or 1
    apl_fraction.append(counts["promyelocyte_abnormal"]/total)
    apl_label.append(int(patient_label[patient] == "PML_RARA"))
if sum(apl_label) >= 3:
    fpr, tpr, cuts = roc_curve(apl_label, apl_fraction)
    resolved["apl_abn_promy"] = float(cuts[np.where(fpr <= 0.05)[0][-1]])
else:
    resolved["apl_abn_promy"] = 0.10   # too few APL patients in dev_fit: keep the floor
print(f"theta_APL = {resolved['apl_abn_promy']:.4f}  ({sum(apl_label)} PML_RARA / {len(apl_label)} AML in dev_fit)")

# --- [REF] reference intervals on CONTROL patients only ----------------------
# Amendment 1b: the population is the AML-MLL controls of dev_fit U dev_cal (~24
# patients), disjoint from train_mil. PBC has no patient identifiers and the reference
# intervals are per-patient proportions, so PBC trains the cell head but not this.
reference_patients = [p for p in (fit_patients + cal_patients) if patient_label[p] == "control"]
control_fractions = {"ig_frac": [], "baso_frac": [], "lymph_frac": [], "smudge_frac": [], "atypical_frac": []}
for patient in reference_patients:
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    total = sum(counts[c] for c in CELL_CLASSES) or 1
    control_fractions["ig_frac"].append((counts["promyelocyte"]+counts["myelocyte"]+counts["metamyelocyte"])/total)
    control_fractions["baso_frac"].append(counts["basophil"]/total)
    control_fractions["lymph_frac"].append(counts["lymphocyte"]/total)
    control_fractions["smudge_frac"].append(counts["smudge_cell"]/total)
    control_fractions["atypical_frac"].append(counts["lymphocyte_atypical"]/total)

percentile = thresholds.reference_percentile
floors = {"ig_cml": ("ig_frac", 0.10), "baso_cml": ("baso_frac", 0.02),
          "lymph_cll": ("lymph_frac", 0.50), "smudge_cll": ("smudge_frac", 0.02),
          "atypical_reactive": ("atypical_frac", 0.10)}
n_controls = len(reference_patients)
reference_report = {}
print(f"{'threshold':<20}{'floor':>8}{'p99':>10}{'adopted':>10}   term")
for name, (quantity, floor) in floors.items():
    p99 = float(np.percentile(control_fractions[quantity], percentile)) if n_controls else 0.0
    resolved[name] = max(floor, p99)
    term = "floor" if floor >= p99 else "empirical"
    reference_report[name] = {"clinical_floor": floor, "empirical_percentile": p99,
                              "n_reference_patients": n_controls, "adopted": resolved[name],
                              "adopted_term": term}
    print(f"{name:<20}{floor:>8.3f}{p99:>10.4f}{resolved[name]:>10.4f}   {term}")
json.dump(reference_report, open(WORK/"results/reference_intervals.json","w"), indent=2)
run.result("reference_intervals.json", reference_report)
print(f"\nreference population: {n_controls} control patients (dev_fit U dev_cal)")
print("Amendment 1b declared that the floor is expected to dominate at this n - report both terms either way.")
# ===== execution [19] =====
# --- blast lineage band, on cell-level pseudo-bags ---------------------------
lymphoblast_index, myeloblast_index = CLASS_INDEX["lymphoblast"], CLASS_INDEX["myeloblast"]
posteriors, truth = [], []
blast_rows = cell_rows[(cell_rows.split=="cell_val") &
                       cell_rows.cell_labels.isin(["lymphoblast","myeloblast"])]
loader = DataLoader(CellDataset(blast_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
with torch.inference_mode():
    for images, allowed, _ in loader:
        probabilities = cell_head(encoder(images.to(DEVICE)))["probabilities"].cpu()
        lymphoid, myeloid = probabilities[:, lymphoblast_index], probabilities[:, myeloblast_index]
        posteriors += (lymphoid/(lymphoid+myeloid+1e-9)).tolist()
        truth += allowed[:, lymphoblast_index].tolist()
fpr, tpr, cuts = roc_curve(truth, posteriors)
eer_index = int(np.argmin(np.abs((1-tpr) - fpr)))
centre = float(cuts[eer_index])
resolved["lineage_lo"], resolved["lineage_hi"] = max(0.0, centre-0.15), min(1.0, centre+0.15)
print(f"lineage EER at {centre:.3f} -> indeterminate band "
      f"[{resolved['lineage_lo']:.2f}, {resolved['lineage_hi']:.2f}]  (n={len(truth)} blasts)")
# ===== execution [20] =====
import yaml
from sklearn.metrics import roc_curve

thresholds = load_thresholds(REPO/"src/aster_block2/decision_grid.yaml")
resolved = {}

# --- tau_abn: specificity >= 0.95 on dev_fit ---------------------------------
mil.eval(); scores, labels = [], []
with torch.inference_mode():
    for patient in fit_patients:
        scores.append(mil(cache[patient][:BAG].to(DEVICE))["probability"].item())
        labels.append(is_positive[patient])
fpr, tpr, cuts = roc_curve(labels, scores)
resolved["p_abn"] = float(cuts[np.where(fpr <= 0.05)[0][-1]])
print(f"tau_abn = {resolved['p_abn']:.4f}  (specificity >= 0.95 on {len(fit_patients)} dev_fit patients)")

# --- theta_APL: specificity >= 0.95 among AML on dev_fit ---------------------
apl_fraction, apl_label = [], []
for patient in fit_patients:
    if patient_label[patient] == "control": continue
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    total = sum(counts[c] for c in CELL_CLASSES) or 1
    apl_fraction.append(counts["promyelocyte_abnormal"]/total)
    apl_label.append(int(patient_label[patient] == "PML_RARA"))
if sum(apl_label) >= 3:
    fpr, tpr, cuts = roc_curve(apl_label, apl_fraction)
    resolved["apl_abn_promy"] = float(cuts[np.where(fpr <= 0.05)[0][-1]])
else:
    resolved["apl_abn_promy"] = 0.10   # too few APL patients in dev_fit: keep the floor
print(f"theta_APL = {resolved['apl_abn_promy']:.4f}  ({sum(apl_label)} PML_RARA / {len(apl_label)} AML in dev_fit)")

# --- [REF] reference intervals on CONTROL patients only ----------------------
# Amendment 1b: the population is the AML-MLL controls of dev_fit U dev_cal (~24
# patients), disjoint from train_mil. PBC has no patient identifiers and the reference
# intervals are per-patient proportions, so PBC trains the cell head but not this.
reference_patients = [p for p in (fit_patients + cal_patients) if patient_label[p] == "control"]
from aster_block2.grid import QUANTITIES as _Q, CELL_CLASSES as _LEUKOCYTES
control_fractions = {"ig_frac": [], "baso_frac": [], "lymph_frac": [], "smudge_frac": [], "atypical_frac": []}
for patient in reference_patients:
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    n_c = sum(counts[c] for c in _LEUKOCYTES) or 1
    for quantity in control_fractions:
        k = sum(counts[c] for c in _Q[quantity])
        corrected, _variance, _ok = quantifier.correct(quantity, k, n_c)
        control_fractions[quantity].append(corrected)

percentile = thresholds.reference_percentile
floors = {"ig_cml": ("ig_frac", 0.10), "baso_cml": ("baso_frac", 0.02),
          "lymph_cll": ("lymph_frac", 0.50), "smudge_cll": ("smudge_frac", 0.02),
          "atypical_reactive": ("atypical_frac", 0.10)}
n_controls = len(reference_patients)
reference_report = {}
print(f"{'threshold':<20}{'floor':>8}{'p99':>10}{'adopted':>10}   term")
for name, (quantity, floor) in floors.items():
    p99 = float(np.percentile(control_fractions[quantity], percentile)) if n_controls else 0.0
    resolved[name] = max(floor, p99)
    term = "floor" if floor >= p99 else "empirical"
    reference_report[name] = {"clinical_floor": floor, "empirical_percentile": p99,
                              "n_reference_patients": n_controls, "adopted": resolved[name],
                              "adopted_term": term}
    print(f"{name:<20}{floor:>8.3f}{p99:>10.4f}{resolved[name]:>10.4f}   {term}")
json.dump(reference_report, open(WORK/"results/reference_intervals.json","w"), indent=2)
run.result("reference_intervals.json", reference_report)
print(f"\nreference population: {n_controls} control patients (dev_fit U dev_cal)")
print("Amendment 1b declared that the floor is expected to dominate at this n - report both terms either way.")
# ===== execution [21] =====
print(sorted(round(x, 3) for x in control_fractions["smudge_frac"]))
# ===== execution [22] =====
# --- blast lineage band, on cell-level pseudo-bags ---------------------------
lymphoblast_index, myeloblast_index = CLASS_INDEX["lymphoblast"], CLASS_INDEX["myeloblast"]
posteriors, truth = [], []
blast_rows = cell_rows[(cell_rows.split=="cell_val") &
                       cell_rows.cell_labels.isin(["lymphoblast","myeloblast"])]
loader = DataLoader(CellDataset(blast_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
with torch.inference_mode():
    for images, allowed, _ in loader:
        probabilities = cell_head(encoder(images.to(DEVICE)))["probabilities"].cpu()
        lymphoid, myeloid = probabilities[:, lymphoblast_index], probabilities[:, myeloblast_index]
        posteriors += (lymphoid/(lymphoid+myeloid+1e-9)).tolist()
        truth += allowed[:, lymphoblast_index].tolist()
fpr, tpr, cuts = roc_curve(truth, posteriors)
eer_index = int(np.argmin(np.abs((1-tpr) - fpr)))
centre = float(cuts[eer_index])
resolved["lineage_lo"], resolved["lineage_hi"] = max(0.0, centre-0.15), min(1.0, centre+0.15)
print(f"lineage EER at {centre:.3f} -> indeterminate band "
      f"[{resolved['lineage_lo']:.2f}, {resolved['lineage_hi']:.2f}]  (n={len(truth)} blasts)")
# ===== execution [23] =====
from PIL import Image
_worst = max(reference_patients, key=lambda p: control_fractions["smudge_frac"][reference_patients.index(p)])
_paths = mll[mll.patient_id == _worst].corpus_path.tolist()
_smudge = []
for _start in range(0, len(_paths), 128):
    _batch = _paths[_start:_start+128]
    _x = torch.stack([eval_tf(Image.open(DATA/"corpus"/p).convert("RGB")) for p in _batch]).to(DEVICE)
    with torch.inference_mode():
        _pred = cell_head(encoder(_x))["logits"].argmax(-1).cpu().tolist()
    _smudge += [p for p, k in zip(_batch, _pred) if CELL_CLASSES[k] == "smudge_cell"]
import matplotlib.pyplot as plt
_fig, _axes = plt.subplots(3, 8, figsize=(16, 6))
for _ax, _p in zip(_axes.flat, _smudge[:24]):
    _ax.imshow(Image.open(DATA/"corpus"/_p)); _ax.axis("off")
plt.suptitle(f"patient témoin {_worst} : {len(_smudge)} cellules prédites Gumprecht"); plt.show()
# ===== execution [24] =====
# --- blast lineage band, on cell-level pseudo-bags ---------------------------
lymphoblast_index, myeloblast_index = CLASS_INDEX["lymphoblast"], CLASS_INDEX["myeloblast"]
posteriors, truth = [], []
blast_rows = cell_rows[(cell_rows.split=="cell_val") &
                       cell_rows.cell_labels.isin(["lymphoblast","myeloblast"])]
loader = DataLoader(CellDataset(blast_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
with torch.inference_mode():
    for images, allowed, _ in loader:
        probabilities = cell_head(encoder(images.to(DEVICE)))["probabilities"].cpu()
        lymphoid, myeloid = probabilities[:, lymphoblast_index], probabilities[:, myeloblast_index]
        posteriors += (lymphoid/(lymphoid+myeloid+1e-9)).tolist()
        truth += allowed[:, lymphoblast_index].tolist()
fpr, tpr, cuts = roc_curve(truth, posteriors)
eer_index = int(np.argmin(np.abs((1-tpr) - fpr)))
centre = float(cuts[eer_index])
resolved["lineage_lo"], resolved["lineage_hi"] = max(0.0, centre-0.15), min(1.0, centre+0.15)
print(f"lineage EER at {centre:.3f} -> indeterminate band "
      f"[{resolved['lineage_lo']:.2f}, {resolved['lineage_hi']:.2f}]  (n={len(truth)} blasts)")
# ===== execution [25] =====
# --- OOD domain gate, N_c/N floor, and MIL temperature -----------------------
from aster_block2.ood import DomainGate
from aster_block2.calibration import TemperatureScaler, expected_calibration_error

# OOD: fitted on in-domain dev bags at a 99% in-domain pass rate  [FIT]
in_domain = torch.cat([cache[p] for p in fit_patients]).numpy()
cal_bags  = [cache[p].numpy() for p in cal_patients]
domain_gate = DomainGate.fit(in_domain, cal_bags, pass_rate=0.99)
resolved["ood_mahalanobis"] = domain_gate.threshold
domain_gate.save(WORK/"checkpoints/ood_stats.npz")
print(f"OOD threshold = {domain_gate.threshold:.1f}  (99th percentile of {len(cal_bags)} in-domain dev_cal bags)")

# N_c/N floor: 1st percentile of the dev distribution  [FIT]
classified_fractions = []
for patient in fit_patients + cal_patients:
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    n_all = sum(counts.values())
    n_leu = sum(counts[c] for c in CELL_CLASSES if c != "other_artifact")
    if n_all: classified_fractions.append(n_leu / n_all)
resolved["min_classified_frac"] = float(np.percentile(classified_fractions, 1))
print(f"N_c/N floor = {resolved['min_classified_frac']:.3f}  "
      f"(1st percentile of {len(classified_fractions)} dev sessions)")

# MIL temperature: fitted on dev_cal, never on train_mil, never on cAItomorph  [FIT]
mil.eval(); cal_scores, cal_labels = [], []
with torch.inference_mode():
    for patient in cal_patients:
        cal_scores.append(mil(cache[patient][:BAG].to(DEVICE))["probability"].item())
        cal_labels.append(is_positive[patient])
calibrator = TemperatureScaler().fit(cal_scores, cal_labels)
resolved["p_abn"] = calibrator.apply(resolved["p_abn"])   # τ on the same scale as the calibrated P_abn
ece_before = expected_calibration_error(cal_scores, cal_labels)
ece_after  = expected_calibration_error([calibrator.apply(s) for s in cal_scores], cal_labels)
resolved["mil_temperature"] = calibrator.temperature
print(f"temperature = {calibrator.temperature:.3f}   ECE {ece_before:.4f} -> {ece_after:.4f}  "
      f"(n={len(cal_patients)} dev_cal patients)")
print("Temperature scaling cannot change the ranking, so AUROC is untouched by construction.")
# ===== execution [26] =====
# --- blast lineage band, on PSEUDO-BAGS (PREREGISTRATION §5.2) ---------------------------
# lineage_post is a SESSION aggregate at inference, so the band is fitted on bags, not cells.
lymphoblast_index, myeloblast_index = CLASS_INDEX["lymphoblast"], CLASS_INDEX["myeloblast"]
blast_rows = cell_rows[(cell_rows.split=="cell_val") & cell_rows.cell_labels.isin(["lymphoblast","myeloblast"])]
loader = DataLoader(CellDataset(blast_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
p_lymph, p_myelo, pred_blast, is_lymph = [], [], [], []
with torch.inference_mode():
    for images, allowed, _ in loader:
        probs = cell_head(encoder(images.to(DEVICE)))["probabilities"].cpu()
        pred = probs.argmax(-1)
        p_lymph += probs[:, lymphoblast_index].tolist(); p_myelo += probs[:, myeloblast_index].tolist()
        pred_blast += ((pred == lymphoblast_index) | (pred == myeloblast_index)).tolist()
        is_lymph += allowed[:, lymphoblast_index].bool().tolist()
p_lymph, p_myelo = np.array(p_lymph), np.array(p_myelo)
pred_blast, is_lymph = np.array(pred_blast), np.array(is_lymph)

def _bag_posteriors(pool, n_bags=1000, bag=50):
    rng = np.random.default_rng(SEED)
    pool = pool[pred_blast[pool]]            # as at inference: only predicted blasts enter
    out = []
    for _ in range(n_bags):
        idx = rng.choice(pool, size=bag, replace=True)
        l, m = p_lymph[idx].sum(), p_myelo[idx].sum()
        out.append(l / (l + m + 1e-9))
    return np.array(out)

lymph_bags = _bag_posteriors(np.where(is_lymph)[0])
myelo_bags = _bag_posteriors(np.where(~is_lymph)[0])
theta_lo = float(np.percentile(lymph_bags, 5))    # <= 5 % of ALL bags called myeloid
theta_hi = float(np.percentile(myelo_bags, 95))   # <= 5 % of AML bags called lymphoid
separable = theta_lo >= theta_hi
if separable:
    theta_lo = theta_hi = (theta_lo + theta_hi) / 2
resolved["lineage_lo"], resolved["lineage_hi"] = theta_lo, theta_hi

print(f"myeloid bags  lineage_post: median {np.median(myelo_bags):.3f}  p95 {np.percentile(myelo_bags,95):.3f}")
print(f"lymphoid bags lineage_post: median {np.median(lymph_bags):.3f}  p5  {np.percentile(lymph_bags,5):.3f}")
print(f"-> {'SEPARABLE, single cut' if separable else 'overlap, indeterminate band'} "
      f"[{theta_lo:.3f}, {theta_hi:.3f}]   (bags of 50 predicted blasts, {is_lymph.sum()} / {(~is_lymph).sum()} cells)")
for name, bags in (("AML bags", myelo_bags), ("ALL bags", lymph_bags)):
    print(f"   {name}: myeloid {np.mean(bags <= theta_lo):.1%}  lymphoid {np.mean(bags >= theta_hi):.1%}  "
          f"indeterminate {np.mean((bags > theta_lo) & (bags < theta_hi)):.1%}")
# ===== execution [27] =====
# --- blast lineage band, on PSEUDO-BAGS (PREREGISTRATION §5.2) ---------------------------
# lineage_post is a SESSION aggregate at inference, so the band is fitted on bags, not cells.
lymphoblast_index, myeloblast_index = CLASS_INDEX["lymphoblast"], CLASS_INDEX["myeloblast"]
blast_rows = cell_rows[(cell_rows.split=="cell_val") & cell_rows.cell_labels.isin(["lymphoblast","myeloblast"])]
loader = DataLoader(CellDataset(blast_rows, DATA/"corpus", eval_tf), batch_size=128, num_workers=2)
p_lymph, p_myelo, pred_blast, is_lymph = [], [], [], []
with torch.inference_mode():
    for images, allowed, _ in loader:
        probs = cell_head(encoder(images.to(DEVICE)))["probabilities"].cpu()
        pred = probs.argmax(-1)
        p_lymph += probs[:, lymphoblast_index].tolist(); p_myelo += probs[:, myeloblast_index].tolist()
        pred_blast += ((pred == lymphoblast_index) | (pred == myeloblast_index)).tolist()
        is_lymph += allowed[:, lymphoblast_index].bool().tolist()
p_lymph, p_myelo = np.array(p_lymph), np.array(p_myelo)
pred_blast, is_lymph = np.array(pred_blast), np.array(is_lymph)

LINEAGE_BAG = int(thresholds.tier_pattern * thresholds.blast_acute)   # 200 [STD] x 20 % [WHO] = 40
def _bag_posteriors(pool, n_bags=1000, bag=LINEAGE_BAG):
    rng = np.random.default_rng(SEED)
    pool = pool[pred_blast[pool]]            # as at inference: only predicted blasts enter
    out = []
    for _ in range(n_bags):
        idx = rng.choice(pool, size=bag, replace=True)
        l, m = p_lymph[idx].sum(), p_myelo[idx].sum()
        out.append(l / (l + m + 1e-9))
    return np.array(out)

lymph_bags = _bag_posteriors(np.where(is_lymph)[0])
myelo_bags = _bag_posteriors(np.where(~is_lymph)[0])
theta_lo = float(np.percentile(lymph_bags, 5))    # <= 5 % of ALL bags called myeloid
theta_hi = float(np.percentile(myelo_bags, 95))   # <= 5 % of AML bags called lymphoid
separable = theta_lo >= theta_hi
if separable:
    theta_lo = theta_hi = (theta_lo + theta_hi) / 2
resolved["lineage_lo"], resolved["lineage_hi"] = theta_lo, theta_hi

print(f"myeloid bags  lineage_post: median {np.median(myelo_bags):.3f}  p95 {np.percentile(myelo_bags,95):.3f}")
print(f"lymphoid bags lineage_post: median {np.median(lymph_bags):.3f}  p5  {np.percentile(lymph_bags,5):.3f}")
print(f"-> {'SEPARABLE, single cut' if separable else 'overlap, indeterminate band'} "
      f"[{theta_lo:.3f}, {theta_hi:.3f}]   (bags of 50 predicted blasts, {is_lymph.sum()} / {(~is_lymph).sum()} cells)")
for name, bags in (("AML bags", myelo_bags), ("ALL bags", lymph_bags)):
    print(f"   {name}: myeloid {np.mean(bags <= theta_lo):.1%}  lymphoid {np.mean(bags >= theta_hi):.1%}  "
          f"indeterminate {np.mean((bags > theta_lo) & (bags < theta_hi)):.1%}")
# ===== execution [28] =====
_la = blast_rows.source.to_numpy() == "leukemiaattr"
_lb = _bag_posteriors(np.where(is_lymph & _la)[0])
_mb = _bag_posteriors(np.where(~is_lymph & _la)[0])
print(f"LeukemiaAttr seul : {(is_lymph & _la).sum()} lymphoblastes / {(~is_lymph & _la).sum()} myéloblastes")
print(f"  sacs myéloïdes  médiane {np.median(_mb):.3f}  p95 {np.percentile(_mb,95):.3f}")
print(f"  sacs lymphoïdes médiane {np.median(_lb):.3f}  p5  {np.percentile(_lb,5):.3f}")
print(f"  à la coupure {theta_hi:.3f} : AML -> myéloïde {np.mean(_mb <= theta_lo):.1%}, "
      f"ALL -> lymphoïde {np.mean(_lb >= theta_hi):.1%}")
# ===== execution [29] =====
# --- OOD domain gate, N_c/N floor, and MIL temperature -----------------------
from aster_block2.ood import DomainGate
from aster_block2.calibration import TemperatureScaler, expected_calibration_error

# OOD: fitted on in-domain dev bags at a 99% in-domain pass rate  [FIT]
in_domain = torch.cat([cache[p] for p in fit_patients]).numpy()
cal_bags  = [cache[p].numpy() for p in cal_patients]
domain_gate = DomainGate.fit(in_domain, cal_bags, pass_rate=0.99)
resolved["ood_mahalanobis"] = domain_gate.threshold
domain_gate.save(WORK/"checkpoints/ood_stats.npz")
print(f"OOD threshold = {domain_gate.threshold:.1f}  (99th percentile of {len(cal_bags)} in-domain dev_cal bags)")

# N_c/N floor: 1st percentile of the dev distribution  [FIT]
classified_fractions = []
for patient in fit_patients + cal_patients:
    counts = predict_counts(mll[mll.patient_id==patient].corpus_path.tolist())
    n_all = sum(counts.values())
    n_leu = sum(counts[c] for c in CELL_CLASSES if c != "other_artifact")
    if n_all: classified_fractions.append(n_leu / n_all)
resolved["min_classified_frac"] = float(np.percentile(classified_fractions, 1))
print(f"N_c/N floor = {resolved['min_classified_frac']:.3f}  "
      f"(1st percentile of {len(classified_fractions)} dev sessions)")

# MIL temperature: fitted on dev_cal, never on train_mil, never on cAItomorph  [FIT]
mil.eval(); cal_scores, cal_labels = [], []
with torch.inference_mode():
    for patient in cal_patients:
        cal_scores.append(mil(cache[patient][:BAG].to(DEVICE))["probability"].item())
        cal_labels.append(is_positive[patient])
calibrator = TemperatureScaler().fit(cal_scores, cal_labels)
resolved["p_abn"] = calibrator.apply(resolved["p_abn"])   # τ on the same scale as the calibrated P_abn
ece_before = expected_calibration_error(cal_scores, cal_labels)
ece_after  = expected_calibration_error([calibrator.apply(s) for s in cal_scores], cal_labels)
resolved["mil_temperature"] = calibrator.temperature
print(f"temperature = {calibrator.temperature:.3f}   ECE {ece_before:.4f} -> {ece_after:.4f}  "
      f"(n={len(cal_patients)} dev_cal patients)")
print("Temperature scaling cannot change the ranking, so AUROC is untouched by construction.")
# ===== execution [30] =====
print(resolved["p_abn"])
# ===== execution [31] =====
# --- write the amendment ------------------------------------------------------
STAMP = time.strftime("%Y-%m-%d")
grid_path = REPO/"src/aster_block2/decision_grid.yaml"
grid = yaml.safe_load(grid_path.read_text())
grid.setdefault("amendments", []).append({
    "date": STAMP,
    "reason": "Phase 1 resolution of the [REF] and [FIT] placeholders",
    "fitted_on": {"FIT": "aml_mll dev_fit patients", "REF": f"{n_controls} aml_mll dev_cal control patients"},
    "never_used": "caitomorph",
    "values": {k: round(v, 6) for k, v in resolved.items()},
})
grid_path.write_text(yaml.safe_dump(grid, sort_keys=False, allow_unicode=True))
json.dump(resolved, open(WORK/"results/resolved_thresholds.json","w"), indent=2)
run.result("resolved_thresholds.json", resolved)

for name, value in resolved.items(): setattr(thresholds, name, value)
missing = thresholds.unresolved()
print("unresolved:", missing if missing else "none - the grid is executable")
print(json.dumps(resolved, indent=2))
# ===== execution [32] =====
# --- pre-flight: is each criterion assertable at all, given the measured error rates? ---
# A threshold sitting near the classifier's own false-positive rate cannot be asserted: the
# correction attributes the observation to error. Better to know it here than to discover
# it as a wall of `indeterminate` in section 9.
from aster_block2.proportion_test import test_proportion

CRITERIA = [("blast_frac", thresholds.blast_acute), ("ig_frac", thresholds.ig_cml_floor),
            ("baso_frac", thresholds.baso_cml_floor), ("lymph_frac", thresholds.lymph_cll_floor),
            ("smudge_frac", thresholds.smudge_cll_floor), ("mono_frac", thresholds.mono_cmml),
            ("abn_promy_frac", 0.10), ("atypical_frac", 0.10)]

print(f"{'criterion':<16}{'theta':>7}{'FPR':>7}   observed fraction needed to assert")
for name, theta in CRITERIA:
    rate = quantifier.rates.get(name)
    fpr = rate.fpr if rate else float("nan")
    needed = None
    for n_c in (200, 500):
        for k in range(0, n_c + 1):
            eff_k, eff_n, ok = quantifier.effective_counts(name, k, n_c)
            if ok and test_proportion(eff_k, eff_n, theta, thresholds.gamma).value == "assert":
                needed = (n_c, k / n_c); break
        if needed and needed[0] == n_c:
            print(f"{name:<16}{theta:>7.2f}{fpr:>7.3f}   N_c={n_c}: {needed[1]:>6.1%}"
                  + ("   <- above 50 %, effectively unassertable" if needed[1] > 0.5 else ""))
            needed = None
        else:
            print(f"{name:<16}{theta:>7.2f}{fpr:>7.3f}   N_c={n_c}: UNREACHABLE"
                  f"   <- this criterion can never fire; report it, do not retune")
# ===== execution [33] =====
# --- tau_abn and temperature, recomputed on the inference bag, without double application ---
from aster_block2.sampling import deterministic_bag
from aster_block2.calibration import TemperatureScaler, expected_calibration_error

def _p_abn_raw(patient):
    feats = cache[patient]
    idx = deterministic_bag(len(feats), BAG, SEED)          # le sac que tire SessionScorer
    with torch.inference_mode():
        return mil(feats[idx].to(DEVICE))["probability"].item()

mil.eval()
_fit_s = [_p_abn_raw(p) for p in fit_patients]; _fit_y = [is_positive[p] for p in fit_patients]
_fpr, _tpr, _cuts = roc_curve(_fit_y, _fit_s)
tau_raw = float(_cuts[np.where(_fpr <= 0.05)[0][-1]])

_cal_s = [_p_abn_raw(p) for p in cal_patients]; _cal_y = [is_positive[p] for p in cal_patients]
calibrator = TemperatureScaler().fit(_cal_s, _cal_y)
resolved["mil_temperature"] = calibrator.temperature
resolved["p_abn"] = calibrator.apply(tau_raw)               # toujours depuis la valeur brute
print(f"tau_abn brut {tau_raw:.4f} -> calibré {resolved['p_abn']:.4f}   T = {calibrator.temperature:.3f}   "
      f"ECE {expected_calibration_error(_cal_s,_cal_y):.4f} -> "
      f"{expected_calibration_error([calibrator.apply(s) for s in _cal_s],_cal_y):.4f}")
# ===== execution [34] =====
# --- write the amendment ------------------------------------------------------
STAMP = time.strftime("%Y-%m-%d")
grid_path = REPO/"src/aster_block2/decision_grid.yaml"
grid = yaml.safe_load(grid_path.read_text())
grid.setdefault("amendments", []).append({
    "date": STAMP,
    "reason": "Phase 1 resolution of the [REF] and [FIT] placeholders",
    "fitted_on": {"FIT": "aml_mll dev_fit patients", "REF": f"{n_controls} aml_mll dev_cal control patients"},
    "never_used": "caitomorph",
    "values": {k: round(v, 6) for k, v in resolved.items()},
})
grid_path.write_text(yaml.safe_dump(grid, sort_keys=False, allow_unicode=True))
json.dump(resolved, open(WORK/"results/resolved_thresholds.json","w"), indent=2)
run.result("resolved_thresholds.json", resolved)

for name, value in resolved.items(): setattr(thresholds, name, value)
missing = thresholds.unresolved()
print("unresolved:", missing if missing else "none - the grid is executable")
print(json.dumps(resolved, indent=2))
# ===== execution [35] =====
# --- pre-flight: is each criterion assertable at all, given the measured error rates? ---
# A threshold sitting near the classifier's own false-positive rate cannot be asserted: the
# correction attributes the observation to error. Better to know it here than to discover
# it as a wall of `indeterminate` in section 9.
from aster_block2.proportion_test import test_proportion

CRITERIA = [("blast_frac", thresholds.blast_acute), ("ig_frac", thresholds.ig_cml_floor),
            ("baso_frac", thresholds.baso_cml_floor), ("lymph_frac", thresholds.lymph_cll_floor),
            ("smudge_frac", thresholds.smudge_cll_floor), ("mono_frac", thresholds.mono_cmml),
            ("abn_promy_frac", 0.10), ("atypical_frac", 0.10)]

print(f"{'criterion':<16}{'theta':>7}{'FPR':>7}   observed fraction needed to assert")
for name, theta in CRITERIA:
    rate = quantifier.rates.get(name)
    fpr = rate.fpr if rate else float("nan")
    needed = None
    for n_c in (200, 500):
        for k in range(0, n_c + 1):
            eff_k, eff_n, ok = quantifier.effective_counts(name, k, n_c)
            if ok and test_proportion(eff_k, eff_n, theta, thresholds.gamma).value == "assert":
                needed = (n_c, k / n_c); break
        if needed and needed[0] == n_c:
            print(f"{name:<16}{theta:>7.2f}{fpr:>7.3f}   N_c={n_c}: {needed[1]:>6.1%}"
                  + ("   <- above 50 %, effectively unassertable" if needed[1] > 0.5 else ""))
            needed = None
        else:
            print(f"{name:<16}{theta:>7.2f}{fpr:>7.3f}   N_c={n_c}: UNREACHABLE"
                  f"   <- this criterion can never fire; report it, do not retune")
# ===== execution [36] =====
# The scorer lives in src/aster_block2/inference.py so that this notebook and the Jetson
# run the SAME code. No second implementation to drift - the same discipline as the
# crop->tensor contract, one level up.
from aster_block2.inference import SessionScorer

scorer = SessionScorer(encoder=encoder, cell_head=cell_head, mil_head=mil,
                       thresholds=thresholds, domain_gate=domain_gate,
                       calibrator=calibrator, quantifier=quantifier,
                       device=DEVICE, bag_size=BAG, seed=SEED)

def score_session(paths, session_id="session", n_fields=0):
    """Thin wrapper keeping the (result, detail) shape used by sections 9-11."""
    result, detail = scorer.score(list(paths), session_id=session_id, number_of_fields=n_fields)
    detail["p_abn"] = result.uncertainty["p_abn"]
    detail["ood"] = result.uncertainty["ood_score"]
    result.label_obj = result          # sections below read .label / .tier / .reasons
    return result, detail

print("session scorer ready (shared with integration/)")
# ===== execution [37] =====
cai = manifest[manifest.source == "caitomorph"].copy()
cai["diagnosis_fine"] = cai.bag_label
records = []
for patient, group in cai.groupby("patient_id"):
    paths = [DATA/"corpus"/p for p in group.corpus_path.tolist()]
    result, detail = score_session(paths, patient)
    records.append({"patient_id": patient, "diagnosis_fine": group.diagnosis_fine.iloc[0],
                    "diagnosis_coarse": json.loads(group.extra.iloc[0]).get("diagnosis_coarse"),
                    "label": result.label, "tier": result.tier,
                    "flags": "|".join(result.flags), "n_classified": result.n_classified,
                    "p_abn": detail["p_abn"], "ood": detail["ood"],
                    "reasons": " ; ".join(result.reasons)})
test_frame = pd.DataFrame(records)
test_frame.to_csv(WORK/"results/caitomorph_409_predictions.csv", index=False)
run.result("caitomorph_409_predictions.csv", test_frame)
print(len(test_frame), "patients scored")
# ===== execution [38] =====
cai = manifest[manifest.source == "caitomorph"].copy()
cai["diagnosis_fine"] = cai.bag_label
records = []
for patient, group in cai.groupby("patient_id"):
    paths = [DATA/"corpus"/p for p in group.corpus_path.tolist()]
    result, detail = score_session(paths, patient)
    records.append({"patient_id": patient, "diagnosis_fine": group.diagnosis_fine.iloc[0],
                    "diagnosis_coarse": json.loads(group.extra.iloc[0]).get("diagnosis_coarse"),
                    "label": result.label, "tier": result.tier,
                    "flags": "|".join(result.flags), "n_classified": result.n_classified,
                    "p_abn": detail["p_abn"], "ood": detail["ood"],
                    "reasons": " ; ".join(result.reasons)})
test_frame = pd.DataFrame(records)
test_frame.to_csv(WORK/"results/caitomorph_409_predictions.csv", index=False)
run.result("caitomorph_409_predictions.csv", test_frame)
print(len(test_frame), "patients scored")
# ===== execution [39] =====
import pickle
cai = manifest[manifest.source == "caitomorph"].copy()
cai["diagnosis_fine"] = cai.bag_label
records, cai_inputs = [], {}
for patient, group in cai.groupby("patient_id"):
    paths = [DATA/"corpus"/p for p in group.corpus_path.tolist()]
    result, detail = score_session(paths, patient)
    # the grid INPUTS, so that §11 replays only the grid, never the encoder
    cai_inputs[patient] = {"counts": detail["counts"], "n_localized": result.number_of_detected_wbc,
                           "p_abn": detail["p_abn"], "lineage_post": detail["lineage_post"],
                           "ood": detail["ood"], "domain_verdict": detail["domain_verdict"]}
    records.append({"patient_id": patient, "diagnosis_fine": group.diagnosis_fine.iloc[0],
                    "diagnosis_coarse": json.loads(group.extra.iloc[0]).get("diagnosis_coarse"),
                    "label": result.label, "tier": result.tier, "flags": "|".join(result.flags),
                    "n_classified": result.number_of_classified_leukocytes,
                    "p_abn": detail["p_abn"], "ood": detail["ood"],
                    "reasons": " ; ".join(result.reasons)})
test_frame = pd.DataFrame(records)
test_frame.to_csv(WORK/"results/caitomorph_409_predictions.csv", index=False)
run.result("caitomorph_409_predictions.csv", test_frame)
pickle.dump(cai_inputs, open(DATA/"cai_inputs.pkl", "wb"))
print(len(test_frame), "patients scored")
print("out_of_domain:", (test_frame.label == "out_of_domain").sum(),
      "| indeterminate:", (test_frame.label == "indeterminate").sum())
# ===== execution [40] =====
print("=== part out_of_domain par diagnostic ===")
print(test_frame.groupby("diagnosis_fine").label
      .apply(lambda s: f"{(s=='out_of_domain').mean():4.0%}  ({(s=='out_of_domain').sum()}/{len(s)})")
      .sort_values(ascending=False).to_string())

def _reason(r):
    if "localizer output not interpretable" in r: return "plancher N_c/N (localiseur)"
    if "cannot be placed with respect to the 20%" in r: return "blastes trop près de 20 %"
    if "cannot separate this population" in r: return "critère non quantifiable"
    if "no rule reached" in r: return "aucune règle assez sûre"
    return r[:70]
ind = test_frame[test_frame.label == "indeterminate"]
print("\n=== raisons des 182 indéterminés ===")
print(ind.reasons.map(_reason).value_counts().to_string())
print("\n=== indéterminés par diagnostic ===")
print(ind.diagnosis_fine.value_counts().to_string())
print("\n=== score OOD médian par diagnostic (seuil 576) ===")
print(test_frame.groupby("diagnosis_fine").ood.median().sort_values().round(0).to_string())
# ===== execution [41] =====
from aster_block2.grid import QUANTITIES as _Q, CELL_CLASSES as _LEU
donors = test_frame[test_frame.diagnosis_fine == "Stem cell donor"].patient_id
rows = []
for p in donors:
    x = cai_inputs[p]; counts = x["counts"]; n = sum(counts.get(c, 0) for c in _LEU)
    def _v(q, theta):
        k = sum(counts.get(c, 0) for c in _Q[q])
        ek, en, ok = quantifier.effective_counts(q, k, n)
        return test_proportion(ek, en, theta, thresholds.gamma).value if ok else "unquantifiable"
    rows.append({"blast<5%": _v("blast_frac", thresholds.blast_normal),
                 "ig<2%": _v("ig_frac", thresholds.ig_normal),
                 "lymph<=50%": _v("lymph_frac", thresholds.lymph_cll),
                 "P_abn<tau": x["p_abn"] < thresholds.p_abn,
                 "raw_blast": sum(counts.get(c, 0) for c in _Q["blast_frac"]) / n,
                 "raw_ig": sum(counts.get(c, 0) for c in _Q["ig_frac"]) / n, "p_abn": x["p_abn"]})
d = pd.DataFrame(rows)
print("R4 exige : reject / reject / reject / True\n")
for col in ("blast<5%", "ig<2%", "lymph<=50%", "P_abn<tau"):
    print(f"  {col:<12}", d[col].value_counts().to_dict())
print(f"\n  blastes bruts   : médiane {d.raw_blast.median():.1%}   p90 {d.raw_blast.quantile(.9):.1%}")
print(f"  gran. immatures : médiane {d.raw_ig.median():.1%}   p90 {d.raw_ig.quantile(.9):.1%}")
print(f"  P_abn           : médiane {d.p_abn.median():.3f}   p90 {d.p_abn.quantile(.9):.3f}   (tau {thresholds.p_abn:.3f})")
ok = (d["blast<5%"]=="reject") & (d["ig<2%"]=="reject") & (d["lymph<=50%"]=="reject") & d["P_abn<tau"]
print(f"\n  donneurs qui passent R4 : {ok.sum()}/{len(d)}")
print("\n=== diagnostics des 25 'blastes trop près de 20 %' ===")
print(test_frame[test_frame.reasons.str.contains("cannot be placed with respect to the 20%")]
      .diagnosis_fine.value_counts().to_string())
# ===== execution [42] =====
_rb = []
for p in [p for p in cache if patient_label[p] == "control"]:
    counts = predict_counts(mll[mll.patient_id == p].corpus_path.tolist())
    n = sum(counts[c] for c in CELL_CLASSES if c != "other_artifact") or 1
    _rb.append(sum(counts[c] for c in ("myeloblast", "lymphoblast", "promyelocyte_abnormal")) / n)
print(f"AML-MLL, {len(_rb)} témoins : blastes prédits  médiane {np.median(_rb):.1%}  "
      f"p90 {np.percentile(_rb, 90):.1%}  max {max(_rb):.1%}")
# ===== execution [43] =====
crosstab = pd.crosstab(test_frame.diagnosis_fine, test_frame.label)
crosstab["n"] = crosstab.sum(1)
crosstab["abstention"] = (test_frame.groupby("diagnosis_fine")
                          .label.apply(lambda s: (s=="indeterminate").mean()).round(3))
print(crosstab.to_string())
crosstab.to_csv(WORK/"results/caitomorph_409_confusion.csv")
run.result("caitomorph_409_confusion.csv", crosstab.reset_index())
print("\ntiers reached:", test_frame.tier.value_counts().to_dict())

# Tier A endpoint: specificity reported separately against donors and against reactive
from sklearn.metrics import confusion_matrix as cm
acute = test_frame.label.str.startswith("acute_blastic")
for negative_group in ("Stem cell donor", "Reactive changes"):
    subset = test_frame[test_frame.diagnosis_fine == negative_group]
    false_positive = acute[subset.index].sum()
    print(f"specificity vs {negative_group:<18} "
          f"{1-false_positive/len(subset):.3f}  ({false_positive} acute calls / {len(subset)})")
for positive_group in ("AML", "ALL", "AL"):
    subset = test_frame[test_frame.diagnosis_fine == positive_group]
    if not len(subset): continue
    print(f"sensitivity (acute call) on {positive_group:<6} "
          f"{acute[subset.index].mean():.3f}  (n={len(subset)})")
# ===== execution [44] =====
# --- section 3.4: secondary analysis restricted to the reference tier ---------
tier_r = test_frame[test_frame.n_classified >= thresholds.tier_reference]
print(f"tier R (N_c >= {thresholds.tier_reference}): {len(tier_r)}/{len(test_frame)} patients")
if len(tier_r):
    print(pd.crosstab(tier_r.diagnosis_fine, tier_r.label).to_string())
    pd.crosstab(tier_r.diagnosis_fine, tier_r.label).to_csv(WORK/"results/caitomorph_tierR_confusion.csv")

# --- falsifier 6.5: tier-P abstention above 50% -------------------------------
pattern_tier = test_frame[test_frame.tier.isin(["pattern", "reference"])]
abstention = (pattern_tier.label == "indeterminate").mean() if len(pattern_tier) else 1.0
print(f"\ntier-P abstention rate: {abstention:.1%}")
if abstention > 0.50:
    run.alert(f"criterion 6.5 MET: tier-P abstention {abstention:.1%} > 50 %. The grid is "
              f"under-powered at the available session sizes; the tier-S screening "
              f"statement becomes the headline claim. Do not retune.", severity="FALSIFIER")
# ===== execution [45] =====
# Tier C, exploratory, wide CI, excluded from the abstract
for label, groups in (("chronic_myeloid_pattern", ["CML","MPN","CMML","MDS / MPN","ET","PV"]),
                      ("chronic_lymphoid_pattern", ["B-cell neoplasm","HCL","T-cell neoplasm","MM"])):
    print(f"\n{label}")
    for group in groups:
        subset = test_frame[test_frame.diagnosis_fine == group]
        if not len(subset): continue
        print(f"  {group:<18} {(subset.label==label).sum():>3}/{len(subset):<3} fired")
    donors = test_frame[test_frame.diagnosis_fine == "Stem cell donor"]
    fired_on_donors = (donors.label == label).sum()
    print(f"  {'Stem cell donor':<18} {fired_on_donors:>3}/{len(donors):<3} fired  <- falsification check 6.5")
# ===== execution [46] =====
from sklearn.metrics import roc_auc_score
HEALTHY, REACTIVE = ["Stem cell donor"], ["Reactive changes"]
NEOPLASMS = [d for d in test_frame.diagnosis_fine.unique() if d not in HEALTHY + REACTIVE]
def _auc(frame, pos, neg):
    s = frame[frame.diagnosis_fine.isin(pos + neg)]
    return roc_auc_score(s.diagnosis_fine.isin(pos).astype(int), s.p_abn), len(s)
inside = test_frame[test_frame.label != "out_of_domain"]
print(f"{'P_abn seul':<44}{'tous':>14}{'hors OOD':>16}")
for name, pos, neg in [("AML vs donneurs", ["AML"], HEALTHY),
                       ("aiguës vs donneurs", ["AML","ALL","AL"], HEALTHY),
                       ("aiguës vs donneurs + réactionnels", ["AML","ALL","AL"], HEALTHY + REACTIVE),
                       ("toute néoplasie vs sains + réactionnels", NEOPLASMS, HEALTHY + REACTIVE)]:
    a, n = _auc(test_frame, pos, neg); b, m = _auc(inside, pos, neg)
    print(f"  {name:<42}{a:>7.3f} (n={n:<3}){b:>8.3f} (n={m})")
aml, don = test_frame[test_frame.diagnosis_fine=="AML"], test_frame[test_frame.diagnosis_fine=="Stem cell donor"]
print(f"\nau seuil tau={thresholds.p_abn:.3f} :  sensibilité AML {(aml.p_abn >= thresholds.p_abn).mean():.3f}"
      f"   spécificité donneurs {(don.p_abn < thresholds.p_abn).mean():.3f}")
print("ancien bloc 2 (MAX, 37 AML / 99 donneurs) : AUROC 0.890, sensibilité 0.676, spécificité 1.000")
# ===== execution [47] =====
import numpy as np
from statsmodels.stats.proportion import proportion_confint
rng = np.random.default_rng(0)
def boot_auc(frame, pos, neg, B=2000):
    s = frame[frame.diagnosis_fine.isin(pos + neg)]
    y = s.diagnosis_fine.isin(pos).to_numpy().astype(int); p = s.p_abn.to_numpy()
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    yb = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
    vals = [roc_auc_score(yb, np.r_[p[rng.choice(ip, len(ip))], p[rng.choice(ineg, len(ineg))]]) for _ in range(B)]
    return roc_auc_score(y, p), *np.percentile(vals, [2.5, 97.5]), len(ip), len(ineg)
for name, pos, neg in [("AML vs donneurs", ["AML"], HEALTHY),
                       ("aiguës vs donneurs + réactionnels", ["AML","ALL","AL"], HEALTHY + REACTIVE)]:
    a, lo, hi, npos, nneg = boot_auc(test_frame, pos, neg)
    print(f"{name:<36} AUROC {a:.3f}  IC95 [{lo:.3f}, {hi:.3f}]  ({npos} vs {nneg})")
def ci(k, n): lo, hi = proportion_confint(k, n, method="wilson"); return f"{k}/{n} = {k/n:.3f} [{lo:.3f}, {hi:.3f}]"
k_sens = int((aml.p_abn >= thresholds.p_abn).sum()); k_spec = int((don.p_abn < thresholds.p_abn).sum())
print(f"\nnouveau, tau figé :  sensibilité {ci(k_sens, len(aml))}   spécificité {ci(k_spec, len(don))}")
print(f"ancien bloc 2     :  sensibilité {ci(25, 37)}   spécificité {ci(99, 99)}")
# ===== execution [48] =====
# --- amendment 9 (0/2): the recalibration module, byte-identical to the Mac repository ---
import hashlib
MODULE = REPO/"src/aster_block2/session_calibration.py"
MODULE.write_text(r'''"""Session-level recalibration of a grid quantity against a manual differential.

Amendment 9 (PREREGISTRATION.md). Post hoc, declared before it was run.

Why the cell-level quantifier is not enough for `blast_frac`. `quantify.py` inverts TPR
and FPR measured on single validation cells from the cell-level sources. Those rates do
not transfer to whole sessions of another acquisition domain: on the AML-MLL controls,
whose manual differential counts ZERO blasts, the corrected blast fraction still sits near
12 %, and the cAItomorph stem-cell donors behave the same way. R4 (`blast_frac < 5 %`)
then cannot fire on normal blood, whatever the rest of the smear looks like.

The correction is the same model as `quantify.py`, fitted one level up. Per patient,

    observed blast fraction = a + b * manual blast fraction + patient-level scatter

with `a` the session-level false-positive rate and `b` = TPR - FPR at session level. It is
fitted on the 189 AML-MLL patients against their manual 100-cell differential. The cell
head never saw AML-MLL (no cell labels), and cAItomorph takes no part.

Uncertainty that reaches the frozen three-way test:
  - binomial counting of the session, inflated by a dispersion factor `phi` that absorbs
    the patient-to-patient scatter of the classifier's error (quasi-binomial);
  - the estimation error of (a, b), from a patient-level bootstrap.
It is turned into an effective (k, n) through the design effect, exactly as
`Quantifier.effective_counts` does, so the frozen test consumes it unchanged.

The correction itself is pure standard library (it runs on the Jetson); only `fit` needs
numpy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .quantify import MIN_SEPARATION, Quantifier


@dataclass
class SessionCalibration:
    """observed = intercept + slope * true, fitted across patients."""

    intercept: float          # a: observed fraction when the manual count is zero
    slope: float              # b: session-level TPR - FPR
    var_intercept: float
    var_slope: float
    covariance: float
    dispersion: float         # phi >= 1, quasi-binomial scatter between patients
    n_patients: int
    diagnostics: dict = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, observed_count: Sequence[int], n_classified: Sequence[int],
            manual_fraction: Sequence[float], strata: Sequence[str],
            manual_total: Sequence[float] | None = None,
            bootstrap: int = 2000, seed: int = 0) -> "SessionCalibration":
        """Ordinary least squares of observed on manual fraction, one row per patient.

        strata        e.g. "control" / "aml": phi is the LARGER of the per-stratum Pearson
                      dispersions, so a stratum that scatters more is never averaged away.
        manual_total  cells in the manual count, only to report the regression-dilution
                      ratio (the manual count is itself a 100-cell sample).
        """
        import numpy as np

        k = np.asarray(observed_count, float)
        n = np.asarray(n_classified, float)
        m = np.asarray(manual_fraction, float)
        strata = np.asarray(strata)
        q = k / n

        def ols(qq, mm):
            slope = np.cov(qq, mm, bias=True)[0, 1] / np.var(mm)
            return qq.mean() - slope * mm.mean(), slope

        a, b = ols(q, m)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            index = rng.integers(0, len(q), len(q))
            if np.var(m[index]) > 0:
                draws.append(ols(q[index], m[index]))
        draws = np.asarray(draws)
        cov = np.cov(draws.T)

        fitted = np.clip(a + b * m, 1e-4, 1 - 1e-4)
        pearson = (q - fitted) ** 2 / (fitted * (1 - fitted) / n)
        dof = len(q) / max(len(q) - 2, 1)
        per_stratum = {str(s): float(pearson[strata == s].mean() * dof) for s in np.unique(strata)}
        phi = max(1.0, *per_stratum.values())

        diagnostics = {
            "slope_ci95": [float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
            "intercept_ci95": [float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
            "dispersion_per_stratum": per_stratum,
            "observed_median_per_stratum": {str(s): float(np.median(q[strata == s]))
                                            for s in np.unique(strata)},
            "n_per_stratum": {str(s): int((strata == s).sum()) for s in np.unique(strata)},
            "bootstrap": len(draws),
        }
        if manual_total is not None:
            t = np.asarray(manual_total, float)
            diagnostics["reliability_ratio"] = float(1 - np.mean(m * (1 - m) / t) / np.var(m))
        return cls(float(a), float(b), float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1]),
                   float(phi), int(len(q)), diagnostics)

    # -- correction (stdlib only) ------------------------------------------
    @property
    def usable(self) -> bool:
        """Pre-declared: the lower 95 % bootstrap bound of the slope clears MIN_SEPARATION."""
        low = self.diagnostics.get("slope_ci95", [self.slope])[0]
        return self.slope >= MIN_SEPARATION and low >= MIN_SEPARATION

    def correct(self, observed_count: int, n: int) -> tuple[float, float, bool]:
        if n <= 0:
            return 0.0, 0.0, False
        observed = observed_count / n
        smoothed = (observed_count + 0.5) / (n + 1)
        if not self.usable:
            return observed, smoothed * (1 - smoothed) / n, False
        corrected = (observed - self.intercept) / self.slope
        # delta method: d/da = -1/b, d/db = -corrected/b ; counting scatter inflated by phi
        variance = (self.dispersion * smoothed * (1 - smoothed) / n
                    + self.var_intercept
                    + corrected ** 2 * self.var_slope
                    + 2 * corrected * self.covariance) / self.slope ** 2
        return min(max(corrected, 0.0), 1.0), max(variance, 1e-12), True

    def effective_counts(self, observed_count: int, n: int) -> tuple[int, int, bool]:
        """Same design-effect construction as Quantifier.effective_counts, capped at n."""
        proportion, variance, ok = self.correct(observed_count, n)
        if not ok:
            return observed_count, n, False
        smoothed = (observed_count + 0.5) / (n + 1)
        variance_raw = smoothed * (1 - smoothed) / n
        n_effective = int(min(n, max(1.0, round(n * variance_raw / variance))))
        return int(round(proportion * n_effective)), n_effective, True


@dataclass
class RecalibratedQuantifier:
    """The cell-level Quantifier, with some grid quantities recalibrated per session.

    Drop-in for `grid.evaluate(quantifier=...)`, which only calls `effective_counts`.
    """

    base: Quantifier
    sessions: dict[str, SessionCalibration] = field(default_factory=dict)

    def effective_counts(self, name: str, observed_count: int, n: int) -> tuple[int, int, bool]:
        if name in self.sessions:
            return self.sessions[name].effective_counts(observed_count, n)
        return self.base.effective_counts(name, observed_count, n)

    def correct(self, name: str, observed_count: int, n: int) -> tuple[float, float, bool]:
        if name in self.sessions:
            return self.sessions[name].correct(observed_count, n)
        return self.base.correct(name, observed_count, n)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(
            {"base": self.base.report(),
             "sessions": {k: asdict(v) for k, v in self.sessions.items()}}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RecalibratedQuantifier":
        from .quantify import ClassRates
        data = json.loads(Path(path).read_text())
        base = Quantifier(rates={k: ClassRates(v["tpr"], v["fpr"], v["n_positive"], v["n_negative"])
                                 for k, v in data["base"].items()})
        return cls(base, {k: SessionCalibration(**v) for k, v in data["sessions"].items()})
''', encoding="utf-8")
digest = hashlib.sha256(MODULE.read_bytes()).hexdigest()
assert digest == "2f646dd9a93d2fcc298d1fd34240f0c2a6dead7093f95690e12f526084a58ee0", f"module differs from the hashed one: {digest}"
print("session_calibration.py OK", digest[:12])
# ===== execution [49] =====
# --- amendment 9 (1/2): session-level blast recalibration on the 189 AML-MLL patients ---
# PREREGISTRATION.md amendment 9. cAItomorph takes no part in this cell.
import numpy as np
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint
from aster_block2.session_calibration import SessionCalibration, RecalibratedQuantifier
from aster_block2.grid import QUANTITIES as _Q, CELL_CLASSES as _LEUKOCYTES
from aster_block2.proportion_test import test_proportion

if "cache" not in globals():
    cache = torch.load(DATA/"features_mll.pt")          # written by section 6
mll["differential"] = mll.extra.map(lambda s: json.loads(s).get("differential", {}))
cell_head.eval()
BLAST = set(_Q["blast_frac"])
rows = []
for patient, group in mll.groupby("patient_id"):
    manual, label = group.differential.iloc[0], group.bag_label.iloc[0]
    total = float(manual.get("pb_total") or 0)
    if not total or patient not in cache: continue
    with torch.inference_mode():   # same argmax over all crops as SessionScorer
        names = [CELL_CLASSES[i] for i in cell_head(cache[patient].to(DEVICE))["logits"].argmax(-1).cpu().tolist()]
    n_c = sum(name in _LEUKOCYTES for name in names)
    k = sum(name in BLAST for name in names)
    # WHO blast equivalents: in APL the counted promyelocytes are the abnormal ones
    blasts = float(manual.get("pb_myeloblast") or 0)
    if label == "PML_RARA": blasts += float(manual.get("pb_promyelocyte") or 0)
    rows.append({"patient_id": patient, "label": label,
                 "stratum": "control" if label == "control" else "aml",
                 "k": k, "n_c": n_c, "q": k / n_c, "manual": blasts / total, "manual_total": total})
cal_frame = pd.DataFrame(rows)

blast_cal = SessionCalibration.fit(cal_frame.k, cal_frame.n_c, cal_frame.manual, cal_frame.stratum,
                                   manual_total=cal_frame.manual_total, seed=SEED)
d = blast_cal.diagnostics
fmt = lambda pair: f"[{pair[0]:.4f}, {pair[1]:.4f}]"
print(f"patients                 {d['n_per_stratum']}")
print(f"observed blast, median   " + "  ".join(f"{s} {v:.3f}" for s, v in d['observed_median_per_stratum'].items()))
print(f"a  session FPR           {blast_cal.intercept:.4f}  IC95 {fmt(d['intercept_ci95'])}")
print(f"b  session TPR - FPR     {blast_cal.slope:.4f}  IC95 {fmt(d['slope_ci95'])}")
print(f"phi dispersion           {blast_cal.dispersion:.2f}   per stratum " + "  ".join(f"{s} {v:.2f}" for s, v in d['dispersion_per_stratum'].items()))
print(f"regression dilution      {d['reliability_ratio']:.3f}   (reported, not corrected)")
print(f"9c-1 applicable          {blast_cal.usable}   (lower IC95 of b >= 0.05)")

recal = RecalibratedQuantifier(quantifier, {"blast_frac": blast_cal})
recal.save(WORK/"checkpoints/quantifier_amendment9.json")
run.result("amendment9_blast_calibration.json", {"calibration": vars(blast_cal), "usable": blast_cal.usable})
run.result("amendment9_mll_patients.csv", cal_frame)

# --- 9c-3: 5-fold patient-level cross-validation on AML-MLL -----------------
def blast_verdicts(q_obj, k, n):
    ke, ne, ok = q_obj.effective_counts("blast_frac", int(k), int(n))
    if not ok: return "indeterminate", "indeterminate"
    return (test_proportion(ke, ne, thresholds.blast_normal, thresholds.gamma).value,
            test_proportion(ke, ne, thresholds.blast_acute, thresholds.gamma).value)
cv_rows = []
for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(cal_frame, cal_frame.label):
    tr = cal_frame.iloc[train_idx]
    fold_q = RecalibratedQuantifier(quantifier, {"blast_frac": SessionCalibration.fit(
        tr.k, tr.n_c, tr.manual, tr.stratum, seed=SEED)})
    for _, r in cal_frame.iloc[test_idx].iterrows():
        (b5, b20), (a5, a20) = blast_verdicts(quantifier, r.k, r.n_c), blast_verdicts(fold_q, r.k, r.n_c)
        cv_rows.append({"patient_id": r.patient_id, "stratum": r.stratum, "manual": r.manual,
                        "ge5_before": b5, "ge20_before": b20, "ge5_after": a5, "ge20_after": a20})
cv = pd.DataFrame(cv_rows)
run.result("amendment9_mll_cv.csv", cv)
def share(series, value):
    k, n = int((series == value).sum()), len(series)
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k}/{n} = {k/n:.2f} [{lo:.2f}, {hi:.2f}]"
ctrl, acute = cv[cv.stratum == "control"], cv[(cv.stratum == "aml") & (cv.manual >= 0.20)]
print(f"\n9c-3  5-fold CV, AML-MLL                  {'cell-level quantifier':<30}  amendment 9")
print(f"controls: blast >= 5 %  REJECTED           {share(ctrl.ge5_before, 'reject'):<30}  {share(ctrl.ge5_after, 'reject')}")
print(f"controls: blast >= 20 % ASSERTED (harm)    {share(ctrl.ge20_before, 'assert'):<30}  {share(ctrl.ge20_after, 'assert')}")
print(f"AML manual >= 20 %: blast >= 20 % ASSERTED {share(acute.ge20_before, 'assert'):<30}  {share(acute.ge20_after, 'assert')}")
# ===== execution [50] =====
# --- amendment 9 (2/2): replay check, then cAItomorph ONCE -------------------------
# Run only after reading the output of (1/2). Nothing below is refitted, whatever it gives.
import pickle
from aster_block2.grid import evaluate
from aster_block2.ood import NOT_VALIDATED, UNKNOWN
if not blast_cal.usable:
    raise SystemExit("9c-1: amendment 9 NOT APPLICABLE (IC95 of b reaches 0.05). Report it; stop here.")
if "cai_inputs" not in globals():
    cai_inputs = pickle.load(open(DATA/"cai_inputs.pkl", "rb"))

def replay(q_obj):   # the SessionScorer grid call, from the stored per-patient inputs
    out = {}
    for patient, x in cai_inputs.items():
        verdict = x["domain_verdict"]
        r = evaluate(x["counts"], thresholds, n_localized=x["n_localized"], p_abn=x["p_abn"],
                     lineage_post=x["lineage_post"],
                     ood_score=x["ood"] if verdict in (None, UNKNOWN) else None, quantifier=q_obj)
        if verdict == NOT_VALIDATED: r.label = "out_of_domain"
        out[patient] = r
    return out

primary = test_frame.set_index("patient_id").label
frozen = replay(quantifier)
mismatch = [p for p, r in frozen.items() if r.label != primary[p]]
assert not mismatch, f"9c-2 FAILED: replay differs from the primary run on {len(mismatch)} patients {mismatch[:5]} - stop"
print(f"9c-2 replay: {len(frozen)}/{len(primary)} primary labels reproduced exactly\n")

amended = replay(recal)
cmp = test_frame[["patient_id", "diagnosis_fine", "label"]].rename(columns={"label": "frozen"})
cmp["amended"] = cmp.patient_id.map(lambda p: amended[p].label)
cmp["amended_reasons"] = cmp.patient_id.map(lambda p: " ; ".join(amended[p].reasons))
run.result("amendment9_caitomorph_409.csv", cmp)

leukaemic = lambda s: s.str.startswith("acute_blastic") | s.str.startswith("chronic_")
def wilson(mask):
    k, n = int(mask.sum()), len(mask)
    lo, hi = proportion_confint(k, n, method="wilson")
    return f"{k}/{n} = {k/n:.3f} [{lo:.3f}, {hi:.3f}]"
don = cmp[cmp.diagnosis_fine == "Stem cell donor"]; rea = cmp[cmp.diagnosis_fine == "Reactive changes"]
acu = cmp[cmp.diagnosis_fine.isin(["AML", "ALL", "AL"])]
metrics = [("non_leukemic, donors", lambda c: don[c] == "non_leukemic"),
           ("non_leukemic, reactive", lambda c: rea[c] == "non_leukemic"),
           ("specificity vs donors (no leukaemic label)", lambda c: ~leukaemic(don[c])),
           ("specificity vs reactive", lambda c: ~leukaemic(rea[c])),
           ("sensitivity acute (AML/ALL/AL -> acute_blastic)", lambda c: acu[c].str.startswith("acute_blastic")),
           ("indeterminate, all patients", lambda c: cmp[c] == "indeterminate")]
report = {}
print(f"{'':<48}{'frozen (primary)':<32}amendment 9 (post hoc)")
for name, mask in metrics:
    report[name] = {"frozen": wilson(mask("frozen")), "amended": wilson(mask("amended"))}
    print(f"{name:<48}{report[name]['frozen']:<32}{report[name]['amended']}")
run.result("amendment9_metrics.json", report)

changed = cmp[cmp.frozen != cmp.amended]
print(f"\n{len(changed)} labels changed:")
print(changed.groupby(["frozen", "amended"]).size().to_string())
harm = cmp[cmp.diagnosis_fine.isin(["Stem cell donor", "Reactive changes"])
           & leukaemic(cmp.amended) & ~leukaemic(cmp.frozen)]
print(f"\ndonors / reactive newly labelled leukaemic: {len(harm)}")
if len(harm):
    print(harm[["patient_id", "diagnosis_fine", "frozen", "amended"]].to_string(index=False))
    run.alert(f"amendment 9: {len(harm)} donor/reactive patients newly labelled leukaemic",
              detail=harm.to_string(index=False))
print("\n", pd.crosstab(cmp.diagnosis_fine, cmp.amended).to_string())
# ===== execution [51] =====
x40_paths = sorted((DATA/"x40_sessions").rglob("crops/*.png"))
print(len(x40_paths), "x40 crops")

stress = []
CHUNK = 200   # split into sessions of 200 crops = the diagnostic differential tier
for start in range(0, len(x40_paths), CHUNK):
    chunk = x40_paths[start:start+CHUNK]
    if len(chunk) < thresholds.tier_screening: continue
    result, detail = score_session(chunk, f"x40_{start//CHUNK:03d}")
    stress.append({"session": f"x40_{start//CHUNK:03d}", "n": len(chunk), "label": result.label,
                   "tier": result.tier, "ood": detail["ood"], "p_abn": detail["p_abn"],
                   "reasons": " ; ".join(result.reasons)})
stress_frame = pd.DataFrame(stress)
stress_frame.to_csv(WORK/"results/x40_stress.csv", index=False)
run.result("x40_stress.csv", stress_frame)
print(stress_frame.to_string())

withheld = stress_frame.label.isin(["out_of_domain","indeterminate","insufficient_evidence"]).mean()
print(f"\nwithheld: {withheld:.1%} of x40 sessions")
print("old block 2 on the same material: AML, 5/5 folds, p=0.986")
if withheld < 1.0:
    run.alert(f"criterion 6.5 MET: {(1-withheld):.1%} of x40 sessions received a confident "
              f"class. The OOD gate failed; block 2 stays scoped out of the deployment "
              f"domain.", severity="FALSIFIER")
# ===== execution [52] =====
sensitivity = []
base_gamma, base_percentile = thresholds.gamma, thresholds.reference_percentile
for gamma in (0.80, 0.90, 0.95):
    thresholds.gamma = gamma
    labels = []
    for patient, group in cai.groupby("patient_id"):
        paths = [DATA/"corpus"/p for p in group.corpus_path.tolist()]
        labels.append(score_session(paths, patient)[0].label)
    frame = pd.Series(labels)
    sensitivity.append({"parameter": "gamma", "value": gamma,
                        "abstention": (frame=="indeterminate").mean(),
                        "acute_calls": frame.str.startswith("acute_blastic").mean()})
    print(sensitivity[-1])
thresholds.gamma = base_gamma
pd.DataFrame(sensitivity).to_csv(WORK/"results/sensitivity_gamma.csv", index=False)
# ===== execution [53] =====
# --- 11 (fast): sensitivity on the three [OPS] values, replayed from cai_inputs ------
# Every arm sees strictly identical grid inputs (no re-encoding): arms differ ONLY by
# the OPS value. The frozen arm must reproduce the 409 primary labels first.
import pickle
from statsmodels.stats.proportion import proportion_confint
from aster_block2.grid import evaluate
from aster_block2.ood import NOT_VALIDATED, UNKNOWN
if "cai_inputs" not in globals():
    cai_inputs = pickle.load(open(DATA/"cai_inputs.pkl", "rb"))
frame0 = test_frame.set_index("patient_id")

def _replay():
    out = {}
    for patient, x in cai_inputs.items():
        verdict = x["domain_verdict"]
        r = evaluate(x["counts"], thresholds, n_localized=x["n_localized"], p_abn=x["p_abn"],
                     lineage_post=x["lineage_post"],
                     ood_score=x["ood"] if verdict in (None, UNKNOWN) else None, quantifier=quantifier)
        if verdict == NOT_VALIDATED: r.label = "out_of_domain"
        out[patient] = r
    return out

def _summary(results, parameter, value):
    s = pd.DataFrame({"dx": frame0.diagnosis_fine,
                      "label": pd.Series({p: r.label for p, r in results.items()})})
    leuk = s.label.str.startswith("acute_blastic") | s.label.str.startswith("chronic_")
    don, rea = s.dx == "Stem cell donor", s.dx == "Reactive changes"
    acu = s.dx.isin(["AML", "ALL", "AL"])
    chronic = s.label.str.startswith("chronic_")
    return {"parameter": parameter, "value": value,
            "non_leuk_donors": f"{int((s.label[don] == 'non_leukemic').sum())}/{int(don.sum())}",
            "spec_donors": round(float((~leuk[don]).mean()), 3),
            "spec_reactive": round(float((~leuk[rea]).mean()), 3),
            "sens_acute": f"{int(s.label[acu].str.startswith('acute_blastic').sum())}/{int(acu.sum())}",
            "indeterminate": round(float((s.label == "indeterminate").mean()), 3),
            "chronic_calls": int(chronic.sum()),
            "chronic_on_donors": int((chronic & don).sum()),
            "chronic_on_CML_or_B": int((chronic & s.dx.isin(["CML", "B-cell neoplasm"])).sum()),
            "APL_flags": sum("APL_suspicion" in r.flags for r in results.values())}

base_gamma = thresholds.gamma
REF_NAMES = ("ig_cml", "baso_cml", "lymph_cll", "smudge_cll", "atypical_reactive")
base_ref = {name: getattr(thresholds, name) for name in REF_NAMES}

frozen_results = _replay()
mismatch = [p for p, r in frozen_results.items() if r.label != frame0.label[p]]
assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
print(f"frozen arm: {len(frozen_results)}/{len(frame0)} primary labels reproduced\n")

rows = []
try:
    for gamma in (0.80, 0.90, 0.95):
        thresholds.gamma = gamma
        rows.append(_summary(_replay(), "gamma", gamma))
    thresholds.gamma = base_gamma
    if "control_fractions" in globals():
        for percentile in (97.5, 99.0, 99.5):
            for name, (quantity, floor) in floors.items():
                setattr(thresholds, name, max(floor, float(np.percentile(control_fractions[quantity], percentile))))
            rows.append(_summary(_replay(), "reference_percentile", percentile))
    else:
        print("control_fractions not in memory (section 7 not run in this session): percentile arm skipped")
finally:
    thresholds.gamma = base_gamma
    for name, value in base_ref.items(): setattr(thresholds, name, value)

sensitivity_frame = pd.DataFrame(rows)
print(sensitivity_frame.to_string(index=False))
run.result("sensitivity_ops.csv", sensitivity_frame)

# R1a [OPS] - `abn_promy_frac > myeloblast_frac`: dropping it can only matter where the
# APL criterion itself ASSERTS. Count those sessions under the frozen values.
apl_asserts = sum(r.verdicts.get("apl") == "assert" for r in frozen_results.values())
print(f"\nR1a: APL criterion ASSERTED on {apl_asserts} sessions -> "
      + ("dropping the comparison changes nothing (abn_promy_frac unquantifiable)" if apl_asserts == 0
         else "see the dropped arm"))
# ===== execution [54] =====
# --- 11 (fast): sensitivity on the three [OPS] values, replayed from cai_inputs ------
# Every arm sees strictly identical grid inputs (no re-encoding): arms differ ONLY by
# the OPS value. The frozen arm must reproduce the 409 primary labels first.
import pickle
from statsmodels.stats.proportion import proportion_confint
from aster_block2.grid import evaluate
from aster_block2.ood import NOT_VALIDATED, UNKNOWN
if "cai_inputs" not in globals():
    cai_inputs = pickle.load(open(DATA/"cai_inputs.pkl", "rb"))
frame0 = test_frame.set_index("patient_id")

def _replay():
    out = {}
    for patient, x in cai_inputs.items():
        verdict = x["domain_verdict"]
        r = evaluate(x["counts"], thresholds, n_localized=x["n_localized"], p_abn=x["p_abn"],
                     lineage_post=x["lineage_post"],
                     ood_score=x["ood"] if verdict in (None, UNKNOWN) else None, quantifier=quantifier)
        if verdict == NOT_VALIDATED: r.label = "out_of_domain"
        out[patient] = r
    return out

def _summary(results, parameter, value):
    s = pd.DataFrame({"dx": frame0.diagnosis_fine,
                      "label": pd.Series({p: r.label for p, r in results.items()})})
    leuk = s.label.str.startswith("acute_blastic") | s.label.str.startswith("chronic_")
    don, rea = s.dx == "Stem cell donor", s.dx == "Reactive changes"
    acu = s.dx.isin(["AML", "ALL", "AL"])
    chronic = s.label.str.startswith("chronic_")
    return {"parameter": parameter, "value": value,
            "non_leuk_donors": f"{int((s.label[don] == 'non_leukemic').sum())}/{int(don.sum())}",
            "spec_donors": round(float((~leuk[don]).mean()), 3),
            "spec_reactive": round(float((~leuk[rea]).mean()), 3),
            "sens_acute": f"{int(s.label[acu].str.startswith('acute_blastic').sum())}/{int(acu.sum())}",
            "indeterminate": round(float((s.label == "indeterminate").mean()), 3),
            "chronic_calls": int(chronic.sum()),
            "chronic_on_donors": int((chronic & don).sum()),
            "chronic_on_CML_or_B": int((chronic & s.dx.isin(["CML", "B-cell neoplasm"])).sum()),
            "APL_flags": sum("APL_suspicion" in r.flags for r in results.values())}

base_gamma = thresholds.gamma
REF_NAMES = ("ig_cml", "baso_cml", "lymph_cll", "smudge_cll", "atypical_reactive")
base_ref = {name: getattr(thresholds, name) for name in REF_NAMES}

frozen_results = _replay()
mismatch = [p for p, r in frozen_results.items() if r.label != frame0.label[p]]
assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
print(f"frozen arm: {len(frozen_results)}/{len(frame0)} primary labels reproduced\n")

rows = []
try:
    for gamma in (0.80, 0.90, 0.95):
        thresholds.gamma = gamma
        rows.append(_summary(_replay(), "gamma", gamma))
    thresholds.gamma = base_gamma
    if "control_fractions" in globals():
        for percentile in (97.5, 99.0, 99.5):
            for name, (quantity, floor) in floors.items():
                setattr(thresholds, name, max(floor, float(np.percentile(control_fractions[quantity], percentile))))
            rows.append(_summary(_replay(), "reference_percentile", percentile))
    else:
        print("control_fractions not in memory (section 7 not run in this session): percentile arm skipped")
finally:
    thresholds.gamma = base_gamma
    for name, value in base_ref.items(): setattr(thresholds, name, value)

sensitivity_frame = pd.DataFrame(rows)
print(sensitivity_frame.to_string(index=False))
run.result("sensitivity_ops.csv", sensitivity_frame)

# R1a [OPS] - `abn_promy_frac > myeloblast_frac`: dropping it can only matter where the
# APL criterion itself ASSERTS. Count those sessions under the frozen values.
apl_asserts = sum(r.verdicts.get("apl") == "assert" for r in frozen_results.values())
print(f"\nR1a: APL criterion ASSERTED on {apl_asserts} sessions -> "
      + ("dropping the comparison changes nothing (abn_promy_frac unquantifiable)" if apl_asserts == 0
         else "see the dropped arm"))
# ===== execution [55] =====
# --- 11 (fast): sensitivity on the three [OPS] values, replayed from cai_inputs ------
# Every arm sees strictly identical grid inputs (no re-encoding): arms differ ONLY by
# the OPS value. The frozen arm must reproduce the 409 primary labels first.
import pickle
from statsmodels.stats.proportion import proportion_confint
from aster_block2.grid import evaluate
from aster_block2.ood import NOT_VALIDATED, UNKNOWN
if "cai_inputs" not in globals():
    cai_inputs = pickle.load(open(DATA/"cai_inputs.pkl", "rb"))
frame0 = test_frame.set_index("patient_id")

def _replay():
    out = {}
    for patient, x in cai_inputs.items():
        verdict = x["domain_verdict"]
        r = evaluate(x["counts"], thresholds, n_localized=x["n_localized"], p_abn=x["p_abn"],
                     lineage_post=x["lineage_post"],
                     ood_score=x["ood"] if verdict in (None, UNKNOWN) else None, quantifier=quantifier)
        if verdict == NOT_VALIDATED: r.label = "out_of_domain"
        out[patient] = r
    return out

def _summary(results, parameter, value):
    s = pd.DataFrame({"dx": frame0.diagnosis_fine,
                      "label": pd.Series({p: r.label for p, r in results.items()})})
    leuk = s.label.str.startswith("acute_blastic") | s.label.str.startswith("chronic_")
    don, rea = s.dx == "Stem cell donor", s.dx == "Reactive changes"
    acu = s.dx.isin(["AML", "ALL", "AL"])
    chronic = s.label.str.startswith("chronic_")
    return {"parameter": parameter, "value": value,
            "non_leuk_donors": f"{int((s.label[don] == 'non_leukemic').sum())}/{int(don.sum())}",
            "spec_donors": round(float((~leuk[don]).mean()), 3),
            "spec_reactive": round(float((~leuk[rea]).mean()), 3),
            "sens_acute": f"{int(s.label[acu].str.startswith('acute_blastic').sum())}/{int(acu.sum())}",
            "indeterminate": round(float((s.label == "indeterminate").mean()), 3),
            "chronic_calls": int(chronic.sum()),
            "chronic_on_donors": int((chronic & don).sum()),
            "chronic_on_CML_or_B": int((chronic & s.dx.isin(["CML", "B-cell neoplasm"])).sum()),
            "APL_flags": sum("APL_suspicion" in r.flags for r in results.values())}

base_gamma = thresholds.gamma
REF_NAMES = ("ig_cml", "baso_cml", "lymph_cll", "smudge_cll", "atypical_reactive")
base_ref = {name: getattr(thresholds, name) for name in REF_NAMES}

frozen_results = _replay()
mismatch = [p for p, r in frozen_results.items() if r.label != frame0.label[p]]
assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
print(f"frozen arm: {len(frozen_results)}/{len(frame0)} primary labels reproduced\n")

rows = []
try:
    for gamma in (0.80, 0.90, 0.95):
        thresholds.gamma = gamma
        rows.append(_summary(_replay(), "gamma", gamma))
    thresholds.gamma = base_gamma
    if "control_fractions" in globals():
        for percentile in (97.5, 99.0, 99.5):
            for name, (quantity, floor) in floors.items():
                setattr(thresholds, name, max(floor, float(np.percentile(control_fractions[quantity], percentile))))
            rows.append(_summary(_replay(), "reference_percentile", percentile))
    else:
        print("control_fractions not in memory (section 7 not run in this session): percentile arm skipped")
finally:
    thresholds.gamma = base_gamma
    for name, value in base_ref.items(): setattr(thresholds, name, value)

sensitivity_frame = pd.DataFrame(rows)
print(sensitivity_frame.to_string(index=False))
run.result("sensitivity_ops.csv", sensitivity_frame)

# R1a [OPS] - `abn_promy_frac > myeloblast_frac`: dropping it can only matter where the
# APL criterion itself ASSERTS. Count those sessions under the frozen values.
apl_asserts = sum(r.verdicts.get("apl") == "assert" for r in frozen_results.values())
print(f"\nR1a: APL criterion ASSERTED on {apl_asserts} sessions -> "
      + ("dropping the comparison changes nothing (abn_promy_frac unquantifiable)" if apl_asserts == 0
         else "see the dropped arm"))
# ===== execution [56] =====
# --- diagnostic: which threshold drifted since the primary run? ---------------------
from aster_block2.grid import load_thresholds
frozen_file = load_thresholds(REPO/"src/aster_block2/decision_grid.yaml")
expected = {"gamma": frozen_file.gamma, "reference_percentile": frozen_file.reference_percentile, **resolved}
drift = {k: (getattr(thresholds, k, None), v) for k, v in expected.items()
         if getattr(thresholds, k, None) is None or abs(getattr(thresholds, k) - v) > 1e-9}
print("drifted (current -> frozen):", {k: f"{c} -> {v}" for k, (c, v) in drift.items()} if drift else "none")
print(pd.DataFrame([(p, frame0.diagnosis_fine[p], frame0.label[p], frozen_results[p].label) for p in mismatch],
                   columns=["patient", "diagnosis", "primary", "replay now"]).to_string(index=False))

for k, (current, value) in drift.items(): setattr(thresholds, k, value)   # back to the frozen state
again = _replay()
still = [p for p, r in again.items() if r.label != frame0.label[p]]
print(f"\nafter restoring: {len(again) - len(still)}/{len(frame0)} primary labels reproduced")
for p in still[:5]:
    print(p, "| primary:", frame0.reasons[p][:150], "\n   | now:", " ; ".join(again[p].reasons)[:150])
# ===== execution [57] =====
# --- 11 (fast): sensitivity on the three [OPS] values, replayed from cai_inputs ------
# Every arm sees strictly identical grid inputs (no re-encoding): arms differ ONLY by
# the OPS value. The frozen arm must reproduce the 409 primary labels first.
import pickle
from statsmodels.stats.proportion import proportion_confint
from aster_block2.grid import evaluate
from aster_block2.ood import NOT_VALIDATED, UNKNOWN
if "cai_inputs" not in globals():
    cai_inputs = pickle.load(open(DATA/"cai_inputs.pkl", "rb"))
frame0 = test_frame.set_index("patient_id")

def _replay():
    out = {}
    for patient, x in cai_inputs.items():
        verdict = x["domain_verdict"]
        r = evaluate(x["counts"], thresholds, n_localized=x["n_localized"], p_abn=x["p_abn"],
                     lineage_post=x["lineage_post"],
                     ood_score=x["ood"] if verdict in (None, UNKNOWN) else None, quantifier=quantifier)
        if verdict == NOT_VALIDATED: r.label = "out_of_domain"
        out[patient] = r
    return out

def _summary(results, parameter, value):
    s = pd.DataFrame({"dx": frame0.diagnosis_fine,
                      "label": pd.Series({p: r.label for p, r in results.items()})})
    leuk = s.label.str.startswith("acute_blastic") | s.label.str.startswith("chronic_")
    don, rea = s.dx == "Stem cell donor", s.dx == "Reactive changes"
    acu = s.dx.isin(["AML", "ALL", "AL"])
    chronic = s.label.str.startswith("chronic_")
    return {"parameter": parameter, "value": value,
            "non_leuk_donors": f"{int((s.label[don] == 'non_leukemic').sum())}/{int(don.sum())}",
            "spec_donors": round(float((~leuk[don]).mean()), 3),
            "spec_reactive": round(float((~leuk[rea]).mean()), 3),
            "sens_acute": f"{int(s.label[acu].str.startswith('acute_blastic').sum())}/{int(acu.sum())}",
            "indeterminate": round(float((s.label == "indeterminate").mean()), 3),
            "chronic_calls": int(chronic.sum()),
            "chronic_on_donors": int((chronic & don).sum()),
            "chronic_on_CML_or_B": int((chronic & s.dx.isin(["CML", "B-cell neoplasm"])).sum()),
            "APL_flags": sum("APL_suspicion" in r.flags for r in results.values())}

base_gamma = thresholds.gamma
REF_NAMES = ("ig_cml", "baso_cml", "lymph_cll", "smudge_cll", "atypical_reactive")
base_ref = {name: getattr(thresholds, name) for name in REF_NAMES}

frozen_results = _replay()
mismatch = [p for p, r in frozen_results.items() if r.label != frame0.label[p]]
assert not mismatch, f"frozen arm differs from the primary run on {len(mismatch)} patients - stop"
print(f"frozen arm: {len(frozen_results)}/{len(frame0)} primary labels reproduced\n")

rows = []
try:
    for gamma in (0.80, 0.90, 0.95):
        thresholds.gamma = gamma
        rows.append(_summary(_replay(), "gamma", gamma))
    thresholds.gamma = base_gamma
    if "control_fractions" in globals():
        for percentile in (97.5, 99.0, 99.5):
            for name, (quantity, floor) in floors.items():
                setattr(thresholds, name, max(floor, float(np.percentile(control_fractions[quantity], percentile))))
            rows.append(_summary(_replay(), "reference_percentile", percentile))
    else:
        print("control_fractions not in memory (section 7 not run in this session): percentile arm skipped")
finally:
    thresholds.gamma = base_gamma
    for name, value in base_ref.items(): setattr(thresholds, name, value)

sensitivity_frame = pd.DataFrame(rows)
print(sensitivity_frame.to_string(index=False))
run.result("sensitivity_ops.csv", sensitivity_frame)

# R1a [OPS] - `abn_promy_frac > myeloblast_frac`: dropping it can only matter where the
# APL criterion itself ASSERTS. Count those sessions under the frozen values.
apl_asserts = sum(r.verdicts.get("apl") == "assert" for r in frozen_results.values())
print(f"\nR1a: APL criterion ASSERTED on {apl_asserts} sessions -> "
      + ("dropping the comparison changes nothing (abn_promy_frac unquantifiable)" if apl_asserts == 0
         else "see the dropped arm"))
# ===== execution [58] =====
# --- 12. Jetson kit: everything the Orin Nano needs, produced from THIS run -------------
# Colab exports the models and a golden set; the engine itself is built ON the Jetson
# (jetson/README_JETSON.md). Nothing here changes a threshold or a result.
import hashlib, shutil
import onnxruntime as ort
from aster_block2.preprocess import build_eval_transform
from aster_block2.sampling import deterministic_bag

assert abs(thresholds.gamma - 0.90) < 1e-12, "thresholds drifted from the frozen state - restore them first"
KIT = run.root/"jetson_kit"; MODELS, GOLDEN = KIT/"models", KIT/"golden"
shutil.rmtree(KIT, ignore_errors=True)
MODELS.mkdir(parents=True); (GOLDEN/"sessions").mkdir(parents=True)
TRT_BATCH = 8   # static batch of the deployed engines (Software_Dev_Micro_Edge export_manifest.json)

# 1. encoder -> ONNX: static batch 8, input 'images', opset 17, classic exporter
encoder.eval().cpu()
dummy = torch.randn(TRT_BATCH, 3, 224, 224)
torch.onnx.export(encoder, dummy, str(MODELS/"encoder.onnx"), input_names=["images"],
                  output_names=["features"], opset_version=17, dynamo=False)
with torch.inference_mode():
    reference = encoder(dummy).numpy()
produced = ort.InferenceSession(str(MODELS/"encoder.onnx"), providers=["CPUExecutionProvider"]).run(
    None, {"images": dummy.numpy()})[0]
onnx_diff = float(np.abs(reference - produced).max())
encoder.to(DEVICE)
print(f"encoder.onnx  {(MODELS/'encoder.onnx').stat().st_size/1e6:.1f} MB   max |torch - onnx| = {onnx_diff:.2e}")
assert onnx_diff < 1e-3, "the ONNX graph does not reproduce the encoder"

# 2. weights (fp32 encoder kept: reference path + fallback), thresholds at full precision
cpu = lambda module: {k: v.detach().cpu() for k, v in module.state_dict().items()}
torch.save({"encoder": cpu(encoder), "cell_head": cpu(cell_head), "mil": cpu(mil)}, MODELS/"heads.pt")
json.dump({k: float(v) for k, v in resolved.items()}, open(MODELS/"thresholds.json", "w"), indent=2)
shutil.copy2(REPO/"src/aster_block2/decision_grid.yaml", MODELS/"decision_grid.yaml")
# quantifier.save() rounds TPR/FPR to 4 decimals for display; the kit needs the exact rates
json.dump({k: {"tpr": r.tpr, "fpr": r.fpr, "n_positive": r.n_positive, "n_negative": r.n_negative}
           for k, r in quantifier.rates.items()}, open(MODELS/"quantifier.json", "w"), indent=2)
domain_gate.save(MODELS/"ood_stats.npz")
json.dump({"bag_size": BAG, "seed": SEED, "image_size": 224, "trt_batch": TRT_BATCH,
           "embedding_dim": 512, "cell_classes": CELL_CLASSES, "gamma": thresholds.gamma,
           "run_id": run.manifest["run_id"], "torch_colab": torch.__version__,
           "onnx_sha256": hashlib.sha256((MODELS/"encoder.onnx").read_bytes()).hexdigest()},
          open(MODELS/"config.json", "w"), indent=2)

# 3. golden sessions: what the Jetson must reproduce, scored NOW by the same scorer
primary = test_frame.set_index("patient_id")
pool = test_frame[test_frame.n_classified >= 150].sort_values("n_classified")
picks = []
for label, k in (("acute_blastic__myeloid_oriented", 2), ("non_leukemic", 2), ("indeterminate", 1)):
    picks += pool[pool.label == label].patient_id.head(k).tolist()
picks += test_frame[test_frame.label == "out_of_domain"].sort_values("n_classified").patient_id.head(1).tolist()
expected = {}
for patient in picks:
    sources = [DATA/"corpus"/p for p in cai[cai.patient_id == patient].corpus_path.tolist()]
    target = GOLDEN/"sessions"/patient; target.mkdir()
    names = [f"{i:04d}_{p.name}" for i, p in enumerate(sources)]     # order = bag order
    for source, name in zip(sources, names): shutil.copy2(source, target/name)
    result, detail = score_session(sources, patient)
    assert result.label == primary.label[patient], f"{patient}: scorer no longer reproduces the primary run"
    expected[patient] = {"crops": names, "label": result.label, "tier": result.tier,
                         "n_classified": result.number_of_classified_leukocytes,
                         "counts": {k: int(v) for k, v in detail["counts"].items()},
                         "p_abn_raw": float(result.uncertainty["p_abn_raw"]),
                         "p_abn": float(result.uncertainty["p_abn"]), "ood": float(detail["ood"]),
                         "lineage_post": None if detail["lineage_post"] is None else float(detail["lineage_post"]),
                         "bag": [int(i) for i in deterministic_bag(len(sources), BAG, SEED)],
                         "verdicts": dict(result.verdicts)}
    print(f"golden {patient:<10} {len(sources):>4} crops  {result.label}")
json.dump(expected, open(GOLDEN/"expected.json", "w"), indent=2)

# 4. encoder parity: 16 preprocessed crops of the first golden session + fp32 outputs
first = picks[0]
transform = build_eval_transform(224)
tensors = torch.stack([transform(Image.open(GOLDEN/"sessions"/first/n).convert("RGB"))
                       for n in expected[first]["crops"][:16]])
with torch.inference_mode():
    feats = encoder(tensors.to(DEVICE))
    probs = cell_head(feats)["probabilities"]
np.savez_compressed(GOLDEN/"encoder_parity.npz", images=tensors.numpy(),
                    features=feats.cpu().numpy(), probabilities=probs.cpu().numpy())

# 5. checksums, verified on the Mac before the kit is assembled
files = sorted(p for p in KIT.rglob("*") if p.is_file())
(KIT/"COLAB_SHA256SUMS.txt").write_text("".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(KIT)}\n" for p in files))
size = sum(p.stat().st_size for p in files) / 1e6
print(f"\njetson_kit: {len(files)} files, {size:.0f} MB  ->  {KIT}")
print("Download this folder from Drive, then on the Mac:  jetson/assemble_kit.sh <downloaded jetson_kit>")
# ===== execution [59] =====
# --- what leaves this notebook -----------------------------------------------
with run.section("13_bundle"):
    encoder.eval().cpu()
    run.checkpoint("cell_encoder", {"encoder": encoder.state_dict(),
                                    "cell_head": cell_head.state_dict()})
    run.checkpoint("mil_head_final", mil.state_dict())
    encoder.to(DEVICE)

    for path in (WORK/"checkpoints/encoder.onnx", WORK/"checkpoints/ood_stats.npz"):
        if path.exists(): run.bundle(path)
    run.bundle(run.root/"checkpoints/cell_encoder_final.pt", "cell_encoder.pt")
    run.bundle(run.root/"checkpoints/mil_head_final_final.pt", "mil_head.pt")
    run.bundle(REPO/"src/aster_block2/decision_grid.yaml")
    run.result("thresholds.json", {k: round(v, 6) for k, v in resolved.items()})
    run.bundle(run.root/"results/thresholds.json")
    run.bundle(REPO/"results/PREREGISTRATION.sha256")
    run.result("bundle_manifest.json", {
        "cell_classes": CELL_CLASSES, "bag_size": BAG, "image_size": 224,
        "grid_sha256_at_freeze": GRID_SHA, "run_id": run.manifest["run_id"],
    })
    run.bundle(run.root/"results/bundle_manifest.json")

run.finalise(
    "Download runs/<run_id>/bundle/ and follow integration/PATCH.md. "
    "Build the TensorRT engine ON THE JETSON - engines are not portable. "
    "REPORT_INPUTS.md maps every artifact to the paper section it feeds."
)
print(open(run.root/"REPORT_INPUTS.md").read())
# ===== execution [60] =====

# --- morning summary: the one thing to read after an unattended run --------------------
from pathlib import Path
import json as _json

manifest = _json.loads((run.root/"manifest.json").read_text())
alerts = (run.root/"ALERTS.md")
print("=" * 72)
print(f"RUN {manifest['run_id']}   started {manifest['started'][:19]}")
print("=" * 72)

if alerts.exists():
    print(alerts.read_text())
else:
    print("\nNo alerts raised.\n")

for name in ("cell_head_training.json", "mil_head_training.json"):
    path = run.root/"results"/name
    if path.exists():
        print(f"{name}: {_json.loads(path.read_text())}")

path = run.root/"results/differential_spearman.json"
if path.exists():
    rho = _json.loads(path.read_text())
    print(f"\nSpearman rho, myeloblast: {rho.get('pb_myeloblast')}")

path = run.root/"results/caitomorph_409_predictions.csv"
if path.exists():
    import pandas as _pd
    frame = _pd.read_csv(path)
    print(f"\ncAItomorph: {len(frame)} patients")
    print(frame.label.value_counts().to_string())

path = run.root/"results/x40_stress.csv"
if path.exists():
    import pandas as _pd
    frame = _pd.read_csv(path)
    withheld = frame.label.isin(["out_of_domain","indeterminate","insufficient_evidence"]).mean()
    print(f"\nx40 stress: {len(frame)} sessions, {withheld:.0%} withheld "
          f"(old block 2 on the same material: AML 5/5, p=0.986)")

print(f"\nEverything is in {run.root}")
print("Read ALERTS.md first, then REPORT_INPUTS.md.")
# ===== execution [61] =====
with open(run.root/"ALERTS.md", "a", encoding="utf-8") as f:
    f.write("""
## RESOLUTIONS — reviewed 2026-09-11

- 07:34 x40 unreadable field: expected; excluded and named; stress test ran on the 614 crops of the remaining fields.
- 10:51 abn_promy_frac unquantifiable: declared limitation; R1a returns indeterminate by construction; reported, not retuned.
- 12:34, 12:35 AttributeError n_classified (section 9): notebook bug (SessionResult names it number_of_classified_leukocytes); fixed, section 9 re-run, 409/409 patients scored after the fix.
- 15:19 (logged x3) frozen arm differs on 9 patients (section 11): an interrupted legacy section-11 cell had left gamma at 0.80. The assertion stopped the cell before any result was written; gamma restored to 0.90, replay 409/409, section 11 re-run. No reported result was computed with the drifted value.
""")
print("ALERTS.md annotated\n")
import glob
path = sorted(glob.glob(str(run.root/"metrics"/"mil*")))[0]
for line in open(path):
    r = json.loads(line)
    print(f"epoch {r.get('epoch'):>3}  train {r.get('loss'):.4f}  val AUROC {r.get('mil_val_auroc'):.4f}  val loss {r.get('mil_val_loss'):.4f}")
# ===== execution [62] =====
with open(run.root/"ALERTS.md", "a", encoding="utf-8") as f:
    f.write("""
## RESOLUTIONS — reviewed 2026-09-11

- 07:34 x40 unreadable field: expected; excluded and named; stress test ran on the 614 crops of the remaining fields.
- 10:51 abn_promy_frac unquantifiable: declared limitation; R1a returns indeterminate by construction; reported, not retuned.
- 12:34, 12:35 AttributeError n_classified (section 9): notebook bug (SessionResult names it number_of_classified_leukocytes); fixed, section 9 re-run, 409/409 patients scored after the fix.
- 15:19 (logged x3) frozen arm differs on 9 patients (section 11): an interrupted legacy section-11 cell had left gamma at 0.80. The assertion stopped the cell before any result was written; gamma restored to 0.90, replay 409/409, section 11 re-run. No reported result was computed with the drifted value.
""")
print("ALERTS.md annotated\n")
import glob
path = sorted(glob.glob(str(run.root/"metrics"/"mil*")))[0]
for line in open(path):
    r = json.loads(line)
    print(f"epoch {r.get('epoch'):>3}  train {r.get('loss'):.4f}  val AUROC {r.get('mil_val_auroc'):.4f}  val loss {r.get('mil_val_loss'):.4f}")
# ===== execution [63] =====
import json
path = run.root/"metrics"/"mil_head.jsonl"
print(path.name, "\n")
fmt = lambda v, f: format(v, f) if isinstance(v, (int, float)) else str(v)
for line in open(path):
    r = json.loads(line)
    if "epoch" not in r:
        print("  (other record)", {k: v for k, v in r.items() if k != "t"}); continue
    print(f"epoch {fmt(r.get('epoch'), '>3')}  train {fmt(r.get('loss'), '.4f')}  "
          f"val AUROC {fmt(r.get('mil_val_auroc'), '.4f')}  val loss {fmt(r.get('mil_val_loss'), '.4f')}")
# ===== execution [64] =====
# --- 15. Archive: everything this session produced, and how, into the Drive run directory ---
# One cell, run last. Each step is independent: a failure is reported, the others still run.
import json, shutil, subprocess, hashlib, time, sys, platform
from pathlib import Path
HIST = run.root/"history"; HIST.mkdir(exist_ok=True)
status = []
def step(name, fn):
    t0 = time.time()
    try:
        status.append((name, "OK", str(fn() or ""), time.time() - t0))
    except Exception as exc:
        status.append((name, "FAILED", f"{type(exc).__name__}: {exc}"[:200], time.time() - t0))

# 1. the notebook exactly as executed: every cell, every output, execution counts
def save_notebook():
    from google.colab import _message
    nb = _message.blocking_request("get_ipynb", request="", timeout_sec=300)["ipynb"]
    path = HIST/"ASTER_block2_executed.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    return f"{len(nb.get('cells', []))} cells, {sum(1 for c in code if c.get('execution_count'))} executed, {path.stat().st_size/1e6:.1f} MB"
step("executed notebook (.ipynb)", save_notebook)

# 2. readable copies of it: HTML to browse, Markdown to quote in the report
def convert(fmt):
    def _run():
        out = subprocess.run([sys.executable, "-m", "nbconvert", "--to", fmt,
                              str(HIST/"ASTER_block2_executed.ipynb"), "--output-dir", str(HIST)],
                             capture_output=True, text=True)
        if out.returncode: raise RuntimeError(out.stderr.strip()[-200:])
        return fmt
    return _run
step("executed notebook (.html)", convert("html"))
step("executed notebook (.md)", convert("markdown"))

# 3. the kernel's own record: every submission since the kernel started, re-runs included
def kernel_history():
    ip = get_ipython()
    parts, count = [], 0
    for _session, number, (source, output) in ip.history_manager.get_range(session=0, raw=True, output=True):
        count += 1
        parts.append(f"# ===== execution [{number}] =====\n{source}\n")
        if output:
            parts.append(f"# ----- Out[{number}] -----\n# {str(output)[:2000]}\n")
    (HIST/"kernel_history.py").write_text("".join(parts), encoding="utf-8")
    hist_file = Path(str(ip.history_manager.hist_file))
    if hist_file.exists():
        shutil.copy2(hist_file, HIST/"ipython_history.sqlite")
    return f"{count} executions"
step("kernel execution history", kernel_history)

# 4. environment
def environment():
    (HIST/"pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                      capture_output=True, text=True).stdout)
    try:
        (HIST/"nvidia_smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    except FileNotFoundError:
        pass
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "run_id": run.manifest["run_id"],
            "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (HIST/"environment.json").write_text(json.dumps(info, indent=2))
    return info["gpu"]
step("environment", environment)

# 5. results that earlier cells printed without saving
def screening():
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.proportion import proportion_confint
    H, R, ACUTE = ["Stem cell donor"], ["Reactive changes"], ["AML", "ALL", "AL"]
    neoplasms = [d for d in test_frame.diagnosis_fine.unique() if d not in H + R]
    inside = test_frame[test_frame.label != "out_of_domain"]
    def auc(frame, pos, neg):
        s = frame[frame.diagnosis_fine.isin(pos + neg)]
        return float(roc_auc_score(s.diagnosis_fine.isin(pos).astype(int), s.p_abn)), int(len(s))
    rng = np.random.default_rng(0)       # same seed and call order as the reported cell
    def boot(pos, neg, B=2000):
        s = test_frame[test_frame.diagnosis_fine.isin(pos + neg)]
        y = s.diagnosis_fine.isin(pos).to_numpy().astype(int); p = s.p_abn.to_numpy()
        i_pos, i_neg = np.where(y == 1)[0], np.where(y == 0)[0]
        yb = np.r_[np.ones(len(i_pos)), np.zeros(len(i_neg))]
        v = [roc_auc_score(yb, np.r_[p[rng.choice(i_pos, len(i_pos))], p[rng.choice(i_neg, len(i_neg))]])
             for _ in range(B)]
        return [float(x) for x in np.percentile(v, [2.5, 97.5])]
    def wilson(k, n):
        lo, hi = proportion_confint(k, n, method="wilson")
        return {"k": int(k), "n": int(n), "rate": k / n, "ci95_wilson": [float(lo), float(hi)]}
    comparisons = {"AML vs donors": (["AML"], H), "acute vs donors": (ACUTE, H),
                   "acute vs donors+reactive": (ACUTE, H + R),
                   "any neoplasm vs donors+reactive": (neoplasms, H + R)}
    out = {"score": "calibrated P_abn, MIL head only (tier S screening)", "comparisons": {}}
    for name, (pos, neg) in comparisons.items():
        a, n = auc(test_frame, pos, neg); b, m = auc(inside, pos, neg)
        out["comparisons"][name] = {"auroc_all": a, "n_all": n, "auroc_excluding_ood": b, "n_excluding_ood": m}
    out["comparisons"]["AML vs donors"]["ci95_bootstrap_2000"] = boot(["AML"], H)
    out["comparisons"]["acute vs donors+reactive"]["ci95_bootstrap_2000"] = boot(ACUTE, H + R)
    aml = test_frame[test_frame.diagnosis_fine == "AML"]
    don = test_frame[test_frame.diagnosis_fine == "Stem cell donor"]
    out["operating_point"] = {"tau_calibrated": float(thresholds.p_abn),
                              "sensitivity_AML": wilson(int((aml.p_abn >= thresholds.p_abn).sum()), len(aml)),
                              "specificity_donors": wilson(int((don.p_abn < thresholds.p_abn).sum()), len(don))}
    out["old_block2_same_patients"] = {"auroc": 0.890, "sensitivity": wilson(25, 37),
                                       "specificity": wilson(99, 99),
                                       "source": "deployed block 2 (MAX head, 5 folds), 37 AML / 99 donors"}
    run.result("p_abn_screening.json", out)
    c = out["comparisons"]["AML vs donors"]
    return f"AML vs donors AUROC {c['auroc_all']:.3f} {[round(x, 3) for x in c['ci95_bootstrap_2000']]}"
step("P_abn screening results", screening)

def tables():
    run.result("caitomorph_409_confusion_full.csv",
               pd.crosstab(test_frame.diagnosis_fine, test_frame.label).reset_index())
    run.result("thresholds_final_state.json", dict(vars(thresholds)))
    return f"gamma {thresholds.gamma}"
step("confusion table + threshold state", tables)

# 6. the AML-MLL feature cache: later analyses without re-encoding
def feature_cache():
    source = DATA/"features_mll.pt"
    if not source.exists():
        return "not on local disk - skipped"
    (run.root/"cache").mkdir(exist_ok=True)
    shutil.copy2(source, run.root/"cache/features_mll.pt")
    return f"{source.stat().st_size/1e6:.0f} MB"
step("AML-MLL feature cache", feature_cache)

# 7. inventory of the whole run directory: path, size, sha256 - and where the weights are
def inventory():
    files = sorted(p for p in run.root.rglob("*") if p.is_file()
                   and p.name not in ("RUN_FILES.sha256", "RUN_INVENTORY.md"))
    with open(run.root/"RUN_FILES.sha256", "w") as handle:
        for p in files:
            handle.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(run.root)}\n")
    groups = {}
    for p in files:
        top = p.relative_to(run.root).parts[0] if len(p.relative_to(run.root).parts) > 1 else "(root)"
        groups.setdefault(top, [0, 0]); groups[top][0] += 1; groups[top][1] += p.stat().st_size
    weights = [p for p in files if p.suffix in (".pt", ".onnx", ".npz")]
    lines = [f"# Run {run.manifest['run_id']} — inventory\n", "| folder | files | MB |", "|---|---|---|"]
    lines += [f"| `{k}` | {n} | {s/1e6:.1f} |" for k, (n, s) in sorted(groups.items())]
    lines += ["", "## Model weights and statistics", "", "| file | MB |", "|---|---|"]
    lines += [f"| `{p.relative_to(run.root)}` | {p.stat().st_size/1e6:.1f} |" for p in weights]
    (run.root/"RUN_INVENTORY.md").write_text("\n".join(lines) + "\n")
    return f"{len(files)} files, {sum(s for _, s in groups.values())/1e6:.0f} MB, {len(weights)} weight/stat files"
step("inventory + sha256 of every file", inventory)

print(f"{'step':<36}{'status':<8}{'s':>6}  detail")
for name, state, detail, seconds in status:
    print(f"{name:<36}{state:<8}{seconds:>6.1f}  {detail}")
print(f"\nrun directory: {run.root}")
print(open(run.root/"RUN_INVENTORY.md").read() if (run.root/"RUN_INVENTORY.md").exists() else "")

# 8. make sure Drive has really received every byte before the session is closed
from google.colab import drive
drive.flush_and_unmount()
print("Drive flushed and unmounted - you can close this session and download the run folder.")
