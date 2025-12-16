import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# ✅ Load model directly from HuggingFace (no local files)
# ======================================================
MODEL_REPO = "mathushaf1989/DPM1"

print(f"📁 Loading Dispute Index model from HuggingFace repo: {MODEL_REPO}")

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

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
