from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import spacy
import torch
import hashlib
import time
from duckduckgo_search import DDGS

@app.route("/",methods=["GET"])
def home():
    return{"status":"API Active"}
    
app = Flask(__name__)
CORS(app)

cache = {}
RESULTS_CACHE = {}


print("Loading models...")
text_tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
text_model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
spacy_nlp = spacy.load("en_core_web_sm")
clip_model = SentenceTransformer('clip-ViT-B-32')
print("Models loaded successfully!")

def verify_headline(headline):
    trusted_domains = ["bbc", "reuters", "apnews", "npr", "gov", "edu"]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=3))
        for result in results:
            url = result.get('href', '').lower()
            for domain in trusted_domains:
                if domain in url:
                    source_name = domain.capitalize()
                    if domain == "gov": source_name = "Government"
                    elif domain == "edu": source_name = "Educational"
                    return f"Verified by {source_name}"
        return "Unverified claim"
    except Exception:
        return "Unverified claim"

def analyze_text(text):
   
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = probs[0].tolist()
    label_idx = torch.argmax(probs).item()
    label = "Fake" if label_idx == 1 else "Real"
    
    
    words = text.split()
    explanations = []
    
    for i in range(min(len(words), 50)):
        temp_words = words.copy()
        temp_words.pop(i)
        temp_text = " ".join(temp_words)
        
        t_inputs = text_tokenizer(temp_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            t_outputs = text_model(**t_inputs)
            t_probs = torch.nn.functional.softmax(t_outputs.logits, dim=-1)
            
            impact = confidence_scores[1] - t_probs[0][1].item()
            explanations.append({"word": words[i], "score": round(impact, 4)})

    doc = spacy_nlp(text)
    entities = list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]))
    
    return {
        "label": label,
        "score": round(confidence_scores[label_idx], 2),
        "entities": entities,
        "verification_note": verify_headline(text),
        "explanation": explanations
    }

def verify_image_context(image_path, headline_text):
    image = Image.open(image_path)
    image_embedding = clip_model.encode(image).reshape(1, -1)
    text_embedding = clip_model.encode(headline_text).reshape(1, -1)
    similarity = cosine_similarity(image_embedding, text_embedding)[0][0]
    image_context = "Consistent" if similarity >= 0.25 else "Mismatch"
    return {"image_context": image_context, "similarity_score": float(similarity)}

@app.route('/analyze-full', methods=['POST'])
def analyze_full_endpoint():
    start_time = time.time()
    data = request.json
    text = data.get('text', '')
    headline = data.get('headline', text)
    image_path = data.get('image_path', '')
    
    hash_input = f"{text}{image_path}{headline}"
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()
    
    if cache_key in cache:
        res = cache[cache_key].copy()
        res['cache_hit'] = True
        return jsonify(res)
    
    try:
        text_res = analyze_text(text)
        has_image = bool(image_path)
        img_context, img_sim = None, None
        
        if has_image:
            try:
                img_res = verify_image_context(image_path, headline)
                img_context, img_sim = img_res['image_context'], img_res['similarity_score']
            except: has_image = False

        
        base_score = 80 if text_res['label'] == "Real" else 40
        if has_image and img_context == "Mismatch": base_score -= 20
        
        result = {
            'final_label': text_res['label'],
            'final_trust_score': max(0, min(100, base_score)),
            'explanation': text_res['explanation'],
            'verification_note': text_res['verification_note'],
            'image_context': img_context,
            'similarity_score': img_sim,
            'reason': text_res['verification_note'],
            'cache_hit': False,
            'response_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        cache[cache_key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)