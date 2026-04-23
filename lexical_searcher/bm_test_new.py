import re
import unicodedata
import json
import numpy as np
from rank_bm25 import BM25Okapi

def normalize(text: str) -> str:
    text = text.lower()
    
    # remove accents/diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # lighter normalization (less destructive)
    replacements = {
        "aa": "a",
        "ee": "i",
        "oo": "u",
        "bh": "b",
        "dh": "d",
        # removed "th" → "t" (too destructive)
        "sh": "s",
        "ṣ": "s",
        "ṅ": "n",
        "ñ": "n"
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# Load data
with open('data.json', 'r') as file:
    data = json.load(file)

docs = []
metadata = []

# Better chunking (overlapping windows)
WINDOW = 10
STRIDE = 5

for song in data:
    # include more searchable context
    text = f"{song['Song Name']} {song['Lyrics']}"
    
    words = normalize(text).split()
    
    for i in range(0, len(words) - WINDOW + 1, STRIDE):
        chunk = words[i:i+WINDOW]
        
        if len(chunk) < 5:
            continue
        
        docs.append(chunk)
        metadata.append(song)

# Build BM25 index
bm25 = BM25Okapi(docs)


def search_bm25(query, top_k=3):
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
        
        # 🔥 phrase match boost
        phrase_bonus = 5 if norm_query in doc_text else 0
        
        score = scores[idx] + phrase_bonus
        
        seen.add(key)
        
        results.append((score, song))
    
    results.sort(reverse=True, key=lambda x: x[0])
    
    return [
        {
            "song": r[1]["Song Name"],
            "composer": r[1]["Composer"],
            "score": float(r[0])
        }
        for r in results[:top_k]
    ]


# Example
print(search_bm25("maha ganapitam manasa swarami"))