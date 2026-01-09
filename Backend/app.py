from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import time
import hashlib
from PIL import Image
import spacy
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)

CACHE = {}


print("Loading models...")

text_tokenizer = AutoTokenizer.from_pretrained(
    "hamzab/roberta-fake-news-classification"
)
text_model = AutoModelForSequenceClassification.from_pretrained(
    "hamzab/roberta-fake-news-classification"
)

spacy_nlp = spacy.load("en_core_web_sm")

clip_model = SentenceTransformer("clip-ViT-B-32")

print("Models loaded successfully!")



@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API Active", "models_loaded": True})


def verify_headline(headline: str) -> str:
    trusted_domains = ["bbc", "reuters", "apnews", "npr", "gov", "edu"]

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=3))

        for r in results:
            url = r.get("href", "").lower()
            for domain in trusted_domains:
                if domain in url:
                    if domain == "gov":
                        return "Verified by Government"
                    if domain == "edu":
                        return "Verified by Educational Institution"
                    return f"Verified by {domain.upper()}"

        return "Unverified claim"

    except Exception:
        return "Unverified claim"


def analyze_text(text: str):
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    label_idx = torch.argmax(probs).item()
    label = "Fake" if label_idx == 1 else "Real"
    confidence = round(probs[label_idx].item(), 2)

   
    doc = spacy_nlp(text)
    entities = list(
        set(
            ent.text
            for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE"]
        )
    )

    words = text.split()[:30]
    explanation = []

    for w in words:
        explanation.append(
            {
                "word": w,
                "score": round(confidence * 0.1, 4),
            }
        )

    return {
        "label": label,
        "confidence": confidence,
        "entities": entities,
        "verification_note": verify_headline(text),
        "explanation": explanation,
    }



def verify_image_context(image_path: str, headline: str):
    image = Image.open(image_path).convert("RGB")

    image_emb = clip_model.encode(image).reshape(1, -1)
    text_emb = clip_model.encode(headline).reshape(1, -1)

    similarity = cosine_similarity(image_emb, text_emb)[0][0]
    context = "Consistent" if similarity >= 0.25 else "Mismatch"

    return context, round(float(similarity), 3)



@app.route("/analyze-full", methods=["POST"])
def analyze_full():
    start_time = time.time()
    data = request.json

    text = data.get("text", "")
    headline = data.get("headline", text)
    image_path = data.get("image_path")

    cache_key = hashlib.md5(
        f"{text}{headline}{image_path}".encode()
    ).hexdigest()

    if cache_key in CACHE:
        cached = CACHE[cache_key].copy()
        cached["cache_hit"] = True
        return jsonify(cached)

 
    text_result = analyze_text(text)

    image_context = None
    similarity_score = None

    if image_path:
        try:
            image_context, similarity_score = verify_image_context(
                image_path, headline
            )
        except Exception:
            image_context = "Image error"

    trust_score = 80 if text_result["label"] == "Real" else 40

    if image_context == "Mismatch":
        trust_score -= 20

    trust_score = max(0, min(100, trust_score))

    response = {
        "final_label": text_result["label"],
        "final_trust_score": trust_score,
        "entities": text_result["entities"],
        "verification_note": text_result["verification_note"],
        "image_context": image_context,
        "similarity_score": similarity_score,
        "explanation": text_result["explanation"],
        "response_time_ms": round((time.time() - start_time) * 1000, 2),
        "cache_hit": False,
    }

    CACHE[cache_key] = response
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
