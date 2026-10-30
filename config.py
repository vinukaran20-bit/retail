"""
config.py
Central configuration for the Retail - Automated Visual Quality Control system.
Edit CLASS_NAMES / paths here if your dataset structure differs.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")   # one sub-folder per class
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")         # optional, same structure
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "vqc_model.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")

# ---- Image / training params ----
IMG_SIZE = (224, 224)      # MobileNetV2 default input
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4

# ---- Classes ----
# Index 0 MUST be the "good" / non-defective class.
CLASS_NAMES = [
    "good",
    "defective",
]

GOOD_CLASS_INDEX = CLASS_NAMES.index("good")

# ---- Human-readable explanations shown to the operator ----
DEFECT_EXPLANATIONS = {
    "defective": "A visual defect was detected on the product surface — it does not match the expected quality standard.",
    "good": "No visual defects were detected. The product matches expected quality standards.",
}
# ======================= FINAL PHASE ADDITIONS (append to config.py) =======================
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")        # unseen pens: test/good, test/defective
EVAL_DIR = os.path.join(BASE_DIR, "static", "evaluation")   # metrics + plots are saved here
DB_PATH = os.path.join(BASE_DIR, "inspection_v2.db")
SECRET_KEY = os.environ.get("VQC_SECRET", "change-this-secret-before-demo")

# Below this confidence the verdict becomes "NEEDS REVIEW" instead of GOOD/DEFECTIVE
REVIEW_THRESHOLD = 0.75

# Email alert (leave SMTP_USER empty to disable). Use environment variables or edit here.
SMTP_HOST = os.environ.get("VQC_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("VQC_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("VQC_SMTP_USER", "")      # e.g. yourname@gmail.com
SMTP_PASS = os.environ.get("VQC_SMTP_PASS", "")      # Gmail "App Password", not your normal password
ALERT_TO = os.environ.get("VQC_ALERT_TO", "")        # who receives the alert

os.makedirs(EVAL_DIR, exist_ok=True)
