import re
import unicodedata
import json
import numpy as np
from rank_bm25 import BM25Okapi
from openai import OpenAI
import os
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

# -----------------------
# Normalization
# -----------------------
def normalize(text: str) -> str:
    text = text.lower()
    
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    replacements = {
        "aa": "a",
        "ee": "i",
        "oo": "u",
        "bh": "b",
        "dh": "d",
        "sh": "s",
        "ṣ": "s",
        "ṅ": "n",
        "ñ": "n"
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# -----------------------
# Load Data
# -----------------------
with open('data.json', 'r') as file:
    data = json.load(file)

docs = []
metadata = []

WINDOW = 10
STRIDE = 5

for song in data:
    text = f"{song['Song Name']} {song['Composer']} {song['Lyrics']}"
    words = normalize(text).split()
    
    for i in range(0, len(words) - WINDOW + 1, STRIDE):
        chunk = words[i:i+WINDOW]
        
        if len(chunk) < 5:
            continue
        
        docs.append(chunk)
        metadata.append(song)

bm25 = BM25Okapi(docs)


# -----------------------
# BM25 Search
# -----------------------
def search_bm25(query, top_k=10):
    norm_query = normalize(query)
    query_tokens = norm_query.split()
    
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:20]
    
    results = []
    seen = set()
    
    for idx in top_indices:
        song = metadata[idx]
        key = (song["Song Name"], song["Composer"])
        
        if key in seen:
            continue
        
        doc_text = " ".join(docs[idx])
        phrase_bonus = 5 if norm_query in doc_text else 0
        score = scores[idx] + phrase_bonus
        
        seen.add(key)
        
        results.append({
            "song": song["Song Name"],
            "composer": song["Composer"],
            "lyrics": song["Lyrics"],
            "score": float(score)
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# -----------------------
# LLM Reranker (Top 3 Selector)
# -----------------------
def pick_top_3_with_llm(query, candidates):
    """
    candidates: list of 10 songs from BM25
    returns: best 3 chosen by Gemini
    """

    prompt = f"""
You are an expert in Carnatic music.

User query:
"{query}"

Below are 10 candidate songs. Each includes title, composer, and lyrics.

Select the TOP 3 most relevant songs.

Criteria:
- phonetic similarity (very important)
- lyric match
- exact kriti recognition

Return ONLY a JSON array of 3 indices (1-based), like:
[2, 5, 1]

Do not include any explanation.

Candidates:
"""

    for i, c in enumerate(candidates):
        lyrics_snippet = c["lyrics"][:300].replace("\n", " ")
        
        prompt += f"""
{i+1}.
Title: {c['song']}
Composer: {c['composer']}
Lyrics: {lyrics_snippet}
"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0
            }
        )

        text = response.text.strip()

        # 🔥 Gemini sometimes wraps JSON in ```json blocks
        text = text.replace("```json", "").replace("```", "").strip()

        indices = json.loads(text)

        selected = []
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                selected.append(candidates[idx - 1])

        if len(selected) == 3:
            return selected

    except Exception as e:
        print("LLM rerank failed:", e)

    # fallback
    return candidates[:3]


# -----------------------
# Final Search Pipeline
# -----------------------
def search(query, top_k=3):
    initial = search_bm25(query, top_k=10)
    final = pick_top_3_with_llm(query, initial)
    return final[:top_k]


# -----------------------
# Example
# -----------------------
if __name__ == "__main__":
    results = search("maha ganapitam manasa smarami")
    
    for r in results:
        print(f"{r['song']} - {r['composer']} (score: {r['score']:.2f})")