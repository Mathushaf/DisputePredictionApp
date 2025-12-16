import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# 🚀 HuggingFace Model Repo
# ======================================================
MODEL_REPO = "mathushaf1989/DPM2"

print(f"📁 Dispute Classification model ready to load from HuggingFace repo: {MODEL_REPO}")

# Lazy-loaded globals
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# 🔄 Lazy Load Function (called only once)
# ======================================================
def _load_model():
    """Loads tokenizer and model only when first needed."""
    global _tokenizer, _model

    if _model is None:
        print("⏳ Loading Dispute Classification Model... (lazy load)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        _model.to(_device)
        _model.eval()
        print("✅ Dispute Classification Model loaded successfully!")


# ======================================================
# 🔮 Prediction Function (safe & low memory)
# ======================================================
def predict_dispute(text: str):
    """
    Predict the dispute classification label using the DPM2 model.
    """

    # Load model on first call
    _load_model()

    # Tokenize
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred_idx = int(torch.argmax(probs, dim=-1).cpu().item())

    return _model.config.id2label[pred_idx]
