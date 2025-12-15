import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# ✅ Resolve absolute path to this file’s directory
#    (Development1/DisputeClassificationModel/)
# ======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# ✅ Absolute path to the classification model folder
#    Model folder name: dispute_classification_DistilBERT
# ======================================================
MODEL_DIR = os.path.join(CURRENT_DIR, "dispute_classification_DistilBERT")

print(f"📁 Loading Dispute Classification model from: {MODEL_DIR}")

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
# ✅ Prediction function
# ======================================================
def predict_dispute(text: str):
    """
    Predict whether the text is a dispute category
    using the classification model.
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

    pred_idx = torch.argmax(probs, dim=1).item()

    # Return human-readable label from config
    return model.config.id2label[pred_idx]
