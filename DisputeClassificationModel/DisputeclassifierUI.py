import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# ✅ Load model from HuggingFace (DPM2)
# ======================================================
MODEL_REPO = "mathushaf1989/DPM2"

print(f"📁 Loading Dispute Classification model from HuggingFace repo: {MODEL_REPO}")

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
# ✅ Prediction function
# ======================================================
def predict_dispute(text: str):
    """
    Predict the dispute classification label using the model.
    """

    # Tokenize
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

    pred_idx = int(torch.argmax(probs, dim=-1).cpu().item())

    # Map index to human-readable label
    return model.config.id2label[pred_idx]
