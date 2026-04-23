from rank_bm25 import BM25Okapi
import json
with open('data.json', 'r') as file:
    data = json.load(file)
corpus = []
for item in data:
    temp = ""
    for line in item['Lyrics']:
        temp = temp + line.lower() + " "
    corpus.append(temp.strip())
bm25 = BM25Okapi(corpus)
query = "pallavi abhirAmIm akhila bhuvana rakSakImAshrayE caraNa ibha vadana shrI guruguha jananIm Isha amrta ghaTEshvara mOhinIm abhaya vara pradAyinIm mArkaNDEyAyuSprada yama bhayApahAriNIm vishudda cakra nivAsnIm"
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)
for i in range(len(corpus)):
    print(f"Document {i} score: {doc_scores[i]}")