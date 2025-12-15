import os
import sys
import re
from flask import Flask, render_template, request
from markupsafe import Markup, escape

# ======================================================
# Add project root to system path
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f"📌 PROJECT ROOT ADDED TO PATH: {PROJECT_ROOT}")

# ======================================================
# Import model UIs
# (Each module now handles its own absolute model paths)
# ======================================================
from DisputeIndex.DisputeIndexUI import predict_index
from DisputeClassificationModel.DisputeclassifierUI import predict_dispute
from NERModel.NERmodelUI import predict_analysis    # <-- CORRECT FILE NAME

# ======================================================
# Initialize Flask
# ======================================================
app = Flask(__name__)

# ======================================================
# Index Helper
# ======================================================
def get_index_color(label: int) -> dict:
    mapping = {
        1: ("Very Low", "Minimal likelihood of dispute"),
        2: ("Low", "Low likelihood of dispute"),
        3: ("Moderate", "Moderate likelihood of dispute"),
        4: ("High", "High likelihood of dispute"),
        5: ("Very High", "Very strong likelihood of dispute"),
    }
    if label in mapping:
        return {"color": mapping[label][0], "description": mapping[label][1]}
    return {"color": "Unknown", "description": "Index out of range"}

# ======================================================
# NER Highlight Colors
# ======================================================
ENTITY_COLOR_MAP = {
    "Clause / Reference No": "ent-color-1",
    "Amount Involved": "ent-color-2",
    "Trigger Phrase": "ent-color-3",
    "Party Involved": "ent-color-4",
    "Time Frame": "ent-color-5",
}

# ======================================================
# NER Parsing Helpers
# ======================================================
def extract_entities_flexible(ner_output):
    """Extract entities no matter how the model formats them."""
    if isinstance(ner_output, dict) and "entities" in ner_output:
        ner_output = ner_output["entities"]

    if not isinstance(ner_output, list):
        return []

    entities = []
    for item in ner_output:
        token = item.get("word") or item.get("token") or item.get("text")
        tag = item.get("tag") or item.get("label") or item.get("type")
        if token and tag:
            entities.append({"entity": token.strip(), "label": tag.strip()})

    return entities


def group_entities_by_label(entities):
    grouped = {}
    for ent in entities:
        grouped.setdefault(ent["label"], []).append(ent["entity"])
    return grouped


def clean_incident_html(html):
    if not html:
        return html
    html = re.sub(r'^(<br\s*/?>|\s|\n|\r)+', '', html, flags=re.IGNORECASE)
    html = re.sub(r'(<br\s*/?>\s*){2,}', '<br>', html, flags=re.IGNORECASE)
    return html.strip()


def split_incidents(text: str):
    """CTRL+ENTER marker incident splitter."""
    return [t.strip() for t in text.split("===NEW INCIDENT===") if t.strip()]


def highlight_text(text, entities):
    html = escape(text)
    for ent in sorted(entities, key=lambda x: len(x["entity"]), reverse=True):
        css = ENTITY_COLOR_MAP.get(ent["label"], "ent-color-0")
        pattern = re.escape(ent["entity"])
        repl = rf"<span class='ner-entity {css}'>\g<0></span>"
        html = re.sub(pattern, repl, html, flags=re.IGNORECASE)
    return html

# ======================================================
# Main Route
# ======================================================
@app.route("/", methods=["GET", "POST"])
def home():

    email = {"category": "", "subject": "", "to": "", "description": ""}

    analysis = {
        "overall_score": None,
        "overall_level": None,
        "overall_description": None,
        "incidents": [],
        "incident_hint": "Tip: Press CTRL + ENTER inside the description box to create a new incident."
    }

    if request.method == "POST":
        email["to"] = request.form.get("to", "").strip()
        email["subject"] = request.form.get("subject", "").strip()
        email["description"] = request.form.get("description", "").strip()

        description_text = email["description"]

        if description_text:

            incidents = split_incidents(description_text)

            incident_results = []
            index_scores = []

            for incident in incidents:

                # Run your 3 models
                idx = predict_index(incident)
                cls = predict_dispute(incident)
                ner = predict_analysis(incident)

                # Dispute Index (integer 1–5)
                try:
                    idx_label = int(idx.get("predicted_label", 0))
                except:
                    idx_label = 0

                ents = extract_entities_flexible(ner)
                grouped = group_entities_by_label(ents)

                # Highlight
                highlighted_html = highlight_text(incident, ents)
                cleaned_html = clean_incident_html(highlighted_html)

                # Store per-incident results
                incident_results.append({
                    "text": incident,
                    "highlighted": Markup(cleaned_html),
                    "index_score": idx_label,
                    "classification": cls,
                    "entities": grouped
                })

                index_scores.append(idx_label)

            # Compute overall score (max severity)
            overall_score = max(index_scores) if index_scores else 0
            info = get_index_color(overall_score)

            analysis.update({
                "overall_score": overall_score,
                "overall_level": info["color"],
                "overall_description": info["description"],
                "incidents": incident_results
            })

    return render_template("index.html", email=email, analysis=analysis)

# ======================================================
# Run server
# ======================================================
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=False)

