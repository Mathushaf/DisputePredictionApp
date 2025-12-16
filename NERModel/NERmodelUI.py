# ======================================================
# 🖥️ NER UI — LegalBERT + BiLSTM + CRF (TorchCRF version)
#     Updated for Render using LAZY LOADING
# ======================================================

import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from huggingface_hub import hf_hub_download   # REQUIRED

# ======================================================
# HuggingFace Repo
# ======================================================
HF_REPO = "mathushaf1989/DPM3"
print(f"📁 NER model available: {HF_REPO} (lazy loading enabled)")

# ======================================================
# Global vars for lazy loading
# ======================================================
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ Labels ------------
ENTITY_TYPES = [
    "Clause / Reference No",
    "Amount Involved",
    "Trigger Phrase",
    "Party Involved",
    "Time Frame",
]

label2id = {"O": 0}
i = 1
for ent in ENTITY_TYPES:
    label2id[f"B-{ent}"] = i; i += 1
    label2id[f"I-{ent}"] = i; i += 1

id2label = {v: k for k, v in label2id.items()}


# ======================================================
# Model Definition
# ======================================================
class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, bert_dir, num_labels, lstm_hidden=512, dropout_rate=0.2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_dir)

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(lstm_hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        emissions = self.classifier(self.dropout(lstm_out))
        mask = attention_mask.bool()

        if labels is not None:
            labels = labels.clone()
            labels[labels < 0] = 0
            return -self.crf(emissions, labels, mask=mask)

        return self.crf.decode(emissions, mask=mask)


# ======================================================
# Lazy Loader
# ======================================================
def load_model_once():
    global tokenizer, model

    if model is not None:
        return

    print("⏳ Lazy loading tokenizer + model...")

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, use_fast=True)

    model = BERT_BiLSTM_CRF(
        HF_REPO,
        num_labels=len(label2id),
        lstm_hidden=512,
        dropout_rate=0.2
    )

    print("⬇️ Downloading model.pt...")
    model_path = hf_hub_download(HF_REPO, "model.pt")

    print(f"📦 Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    print(f"✅ NER model ready on {device}")


# ======================================================
# Cleaning + Helper Functions
# ======================================================
def clean_text_for_ner(text: str) -> str:
    t = re.sub(r'<[^>]+>', ' ', text)
    t = re.sub(r'[\r\t]+', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\s+([.,;:!?])', r'\1', t)
    return t.strip()


def merge_adjacent_same_type(entities, text):
    merged, cur = [], None
    for e in sorted(entities, key=lambda x: (x["start"], x["end"])):

        if cur is None:
            cur = e
            continue

        gap = text[cur["end"]:e["start"]]
        if e["type"] == cur["type"] and (not gap or gap.isspace()):
            cur["end"] = e["end"]
            cur["text"] = text[cur["start"]:cur["end"]]
        else:
            merged.append(cur)
            cur = e

    if cur:
        merged.append(cur)

    return merged


# ======================================================
# Time frame expansion
# ======================================================
_NUM_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
_TIME_UNIT = r"(?:day|days|week|weeks|month|months|year|years)"
_QUAL = r"(?:within|up to|around|about|approximately|under|at least|at most|no more than|not more than|not less than)"

def expand_time_frames(text, entities):
    out = []
    for e in entities:
        if e["type"] != "Time Frame":
            out.append(e)
            continue

        start, end = e["start"], e["end"]

        m_right = re.match(rf"\s*{_TIME_UNIT}\b", text[end:], flags=re.IGNORECASE)
        if m_right:
            end += m_right.end()

        left = text[max(0, start - 50):start]
        m = re.search(rf"(?:{_QUAL}\s+)?(?:(?:\d+(?:\.\d+)?)|(?:{_NUM_WORDS}))\s*$", left, flags=re.IGNORECASE)
        if m:
            start -= (len(left) - m.start())

        out.append({
            "type": "Time Frame",
            "start": start,
            "end": end,
            "text": text[start:end],
        })

    return out


# ======================================================
# Amount rule
# ======================================================
AMOUNT_RX = re.compile(r'(?:Rs\.?|USD|AUD|\$)\s?[\d,]+(?:\.\d+)?', re.IGNORECASE)

def add_amount_rule_fallback(text, entities):
    seen = {(e["type"], e["start"], e["end"]) for e in entities}
    for m in AMOUNT_RX.finditer(text):
        s, e = m.start(), m.end()
        if ("Amount Involved", s, e) not in seen:
            entities.append({
                "type": "Amount Involved",
                "start": s,
                "end": e,
                "text": text[s:e]
            })
    return entities


# ======================================================
# Clause reference rules
# ======================================================
GENERIC_CODE_RX = re.compile(
    r'\b(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{3,15}\b'
)

CLAUSE_PATTERNS = [
    re.compile(r'\bClause\s+\d+(?:\.\d+)*\b', re.IGNORECASE),
    re.compile(r'\b(?:RFI|PMI|REF|X|SP|NRV|LAP|WBSM|IFA|IFC)[A-Za-z0-9]*\d+\b', re.IGNORECASE),
    GENERIC_CODE_RX,
]

def add_clause_refs(text, entities):
    for rx in CLAUSE_PATTERNS:
        for m in rx.finditer(text):
            entities.append({
                "type": "Clause / Reference No",
                "start": m.start(),
                "end": m.end(),
                "text": text[m.start():m.end()],
            })
    return entities


STRICT_KEEP = [
    re.compile(r'\bClause\s+\d+(?:\.\d+)*\b', re.IGNORECASE),
    GENERIC_CODE_RX,
    re.compile(r'\b\d+[A-Z]+[A-Za-z0-9]*\b'),
    re.compile(r'\b[A-Z]+\d+[A-Za-z0-9]*\b'),
]

def is_valid_clause_ref(span):
    return any(rx.search(span) for rx in STRICT_KEEP)

def enforce_clause_validity(text, entities):
    return [
        e for e in entities
        if not (e["type"] == "Clause / Reference No" and not is_valid_clause_ref(e["text"]))
    ]


# ======================================================
# Trigger phrase (court)
# ======================================================
COURT_RX = re.compile(r"\bcourts?\b", re.IGNORECASE)

def add_court_highlight(text, entities):
    for m in COURT_RX.finditer(text):
        entities.append({
            "type": "Trigger Phrase",
            "start": m.start(),
            "end": m.end(),
            "text": text[m.start():m.end()]
        })
    return entities


# ======================================================
# Date → Time Frame
# ======================================================
MONTHS = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t\.?|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'

DATE_PATTERNS = [
    re.compile(rf'\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS}\s+\d{{2,4}}\b', re.IGNORECASE),
    re.compile(rf'{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{2,4}})?', re.IGNORECASE),
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
]

def add_dates_as_timeframes(text, entities):
    for rx in DATE_PATTERNS:
        for m in rx.finditer(text):
            entities.append({
                "type": "Time Frame",
                "start": m.start(),
                "end": m.end(),
                "text": text[m.start():m.end()]
            })
    return entities


# ======================================================
# Trigger cleanup
# ======================================================
def drop_single_word_triggers(entities):
    return [
        e for e in entities
        if not (e["type"] == "Trigger Phrase" and len(re.findall(r'\w+', e["text"])) <= 1)
    ]


# ======================================================
# Dedupe
# ======================================================
def dedupe(entities):
    out, seen = [], set()
    for e in entities:
        key = (e["type"], e["start"], e["end"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ======================================================
# 🔍 Core Prediction Function
# ======================================================
def predict_analysis(text: str):

    load_model_once()

    cleaned = clean_text_for_ner(text)

    enc = tokenizer(
        cleaned,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()
    word_ids = enc.word_ids(0)

    with torch.no_grad():
        pred_ids = model(input_ids, attention_mask)[0]

    word_to_labels = {}
    for wid, lid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        word_to_labels.setdefault(wid, []).append(lid)

    word_spans = []
    for wid, lab_ids in sorted(word_to_labels.items()):
        tok_idxs = [i for i, w in enumerate(word_ids) if w == wid]
        start = offsets[tok_idxs[0]][0]
        end   = offsets[tok_idxs[-1]][1]

        labels = [id2label.get(lid, "O") for lid in lab_ids]
        non_o = [l for l in labels if l != "O"]

        if not non_o:
            etype = None
        else:
            b_labels = [l for l in non_o if l.startswith("B-")]
            chosen = b_labels[0] if b_labels else non_o[0]
            etype = chosen[2:] if chosen.startswith(("B-", "I-")) else None

        word_spans.append({"type": etype, "start": start, "end": end})

    entities = []
    cur = None
    for ws in word_spans:
        if ws["type"] is None:
            if cur:
                cur["text"] = cleaned[cur["start"]:cur["end"]]
                entities.append(cur)
                cur = None
            continue

        if cur is None:
            cur = {"type": ws["type"], "start": ws["start"], "end": ws["end"]}
        else:
            gap = cleaned[cur["end"]:ws["start"]]
            if ws["type"] == cur["type"] and (not gap or gap.isspace()):
                cur["end"] = ws["end"]
            else:
                cur["text"] = cleaned[cur["start"]:cur["end"]]
                entities.append(cur)
                cur = {"type": ws["type"], "start": ws["start"], "end": ws["end"]}

    if cur:
        cur["text"] = cleaned[cur["start"]:cur["end"]]
        entities.append(cur)

    # === Post-processing ===
    entities = expand_time_frames(cleaned, entities)
    entities = merge_adjacent_same_type(entities, cleaned)
    entities = add_amount_rule_fallback(cleaned, entities)
    entities = add_clause_refs(cleaned, entities)
    entities = add_dates_as_timeframes(cleaned, entities)
    entities = add_court_highlight(cleaned, entities)
    entities = drop_single_word_triggers(entities)
    entities = enforce_clause_validity(cleaned, entities)

    # clean text
    for e in entities:
        e["text"] = cleaned[e["start"]:e["end"]].strip()

    entities = [e for e in entities if e["text"]]
    entities = dedupe(entities)

    return {"entities": entities}


# ======================================================
# Manual Test
# ======================================================
if __name__ == "__main__":
    sample = (
        "Issue ID P4bWL5 raised on 17/06/2022. "
        "The court reviewed Clause 15.1 and rejected 30mm as reference."
    )
    print(json.dumps(predict_analysis(sample), indent=2, ensure_ascii=False))
