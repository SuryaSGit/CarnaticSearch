import re
import unicodedata
import time
def normalize(text: str) -> str:
    # lowercase
    text = text.lower()
    
    # remove accents/diacritics
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # normalize common Carnatic transliteration quirks
    replacements = {
        "aa": "a",
        "ee": "i",
        "oo": "u",
        "bh": "b",
        "dh": "d",
        "th": "t",
        "sh": "s",
        "ṣ": "s",
        "ṅ": "n",
        "ñ": "n"
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # remove anything not letters or spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
from rank_bm25 import BM25Okapi
import json
with open('data.json', 'r') as file:
    data = json.load(file)
docs = []
metadata = []

docs = []
metadata = []

for song in data:
    text = (song["Lyrics"])
    
    # split into chunks of ~8–12 words
    words = normalize(text).split()
    
    for i in range(0, len(words), 8):
        chunk = words[i:i+8]
        
        if len(chunk) < 4:
            continue
        
        docs.append(chunk)
        metadata.append(song)

bm25 = BM25Okapi(docs)
import numpy as np

from rapidfuzz import fuzz

def search_bm25(query, top_k=1):
    norm_query = normalize(query)
    query_tokens = norm_query.split()
    
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:20]
    
    results = []
    
    for idx in top_indices:
        doc_text = " ".join(docs[idx])
        
        fuzzy = fuzz.partial_ratio(norm_query, doc_text)
        score = 0.7 * scores[idx] + 0.3 * fuzzy
        
        results.append((score, metadata[idx]))
    
    results.sort(reverse=True, key=lambda x: x[0])
    
    return [
        {
            "song": r[1]["Song Name"],
            "composer": r[1]["Composer"]
        }
        for r in results[:top_k]
    ]
print(search_bm25("nIdu "))