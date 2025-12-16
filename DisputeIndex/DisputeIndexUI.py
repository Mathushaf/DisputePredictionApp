import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_REPO = "mathushaf1989/DPM1"

print(f"📁 Dispute Index model ready to load from HuggingFace repo: {MODEL_REPO}")

# Lazy-loaded globals
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    """Loads the tokenizer and model only when needed."""
    global _tokenizer, _model

    if _model is None:
        print("⏳ Loading Dispute Index model... (lazy load)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        _model.to(_device)
        _model.eval()
        print("✅ Dispute Index model loaded successfully!")

def predict_index(text: str):
    """
    Predicts the discrete Dispute Index class (1–5).
    Model outputs 0–4 internally → mapped to 1–5.
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
        pred_id = int(torch.argmax(probs, dim=-1).cpu().item())  # 0–4

    return {"predicted_label": pred_id + 1}
