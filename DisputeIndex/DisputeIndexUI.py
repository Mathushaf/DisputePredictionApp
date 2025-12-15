import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# ✅ Resolve absolute path to this file’s directory
#    (Development1/DisputeIndex/)
# ======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# ✅ Absolute path to the model folder
#    Your model folder is inside the same directory:
#    Development1/DisputeIndex/DisputeIndex_DistilBERT_Base
# ======================================================
MODEL_DIR = os.path.join(CURRENT_DIR, "DisputeIndex_DistilBERT_Base")

print(f"📁 Loading Dispute Index model from: {MODEL_DIR}")

# ======================================================
# ✅ Load tokenizer + model
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ======================================================
# ✅ Device setup
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ======================================================
# ✅ Prediction function (maps 0–4 → 1–5)
# ======================================================
def predict_index(text: str):
    """
    Predicts the discrete Dispute Index class (1–5).
    Internally the model outputs 0–4 → mapped to 1–5.
    """

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).cpu().item())  # 0–4

    # Convert 0–4 → 1–5
    pred_label = pred_id + 1

    return {"predicted_label": pred_label}
