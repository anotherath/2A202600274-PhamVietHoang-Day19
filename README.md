# GraphRAG vs Flat RAG - Tech Company Corpus

Lab Day 19: Xay dung he thong GraphRAG voi Tech Company Corpus

## Cau truc du an

```
day19/
├── corpus.json              # Du lieu van ban 10 cong ty tech
├── triples.json             # Du lieu quan he (head, relation, tail)
├── main.py                  # Pipeline chinh (Flat RAG + GraphRAG + Benchmark)
├── requirements.txt         # Thu vien can cai dat
├── .env.example             # Mau file cau hinh
└── README.md                # Huong dan nay
```

## Cai dat

### 1. Cai dat thu vien

```bash
pip install -r requirements.txt
```

### 2. Cau hinh API key

```bash
# Copy file mau
cp .env.example .env

# Sua file .env, dien OpenAI API key cua ban
```

Noi dung file `.env`:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

### 3. Chay pipeline

```bash
python main.py
```

## Luong hoat dong

```
Input: corpus.json + triples.json
    |
    v
+------------------+    +------------------+
|   Flat RAG       |    |   Graph RAG      |
| - Chia chunks    |    | - Build graph    |
| - Embedding      |    | - Extract entity |
| - Vector search  |    | - Traverse 2-hop |
| - LLM answer     |    | - LLM answer     |
+------------------+    +------------------+
    |                       |
    v                       v
+------------------------------------------+
|         Benchmark 20 cau hoi             |
|  - So sanh do chinh xac                  |
|  - So sanh token / thoi gian             |
|  - Phat hien hallucination               |
+------------------------------------------+
    |
    v
Output: graph_visualization.png
        benchmark_results.json
```

## Giai thich 2 phuong phap

### Flat RAG
1. Chia van ban thanh cac doan nho (chunks)
2. Tao embedding cho moi chunk bang OpenAI API
3. Khi co cau hoi: tao embedding query -> tim cac chunk gan nhat (cosine similarity)
4. Dua cac chunk tim duoc vao prompt -> LLM tra loi

### Graph RAG
1. Xay dung Knowledge Graph tu cac triples (head, relation, tail)
2. Khi co cau hoi: dung LLM trich xuat entity tu cau hoi
3. Duyet do thi tu entity (BFS, toi da 2 hop)
4. Lay cac quan he tim duoc -> LLM tra loi

## Ket qua benchmark

Sau khi chay, ban se co:

| Output | Mo ta |
|--------|-------|
| `graph_visualization.png` | Hinh anh do thi Knowledge Graph |
| `benchmark_results.json` | Ket qua chi tiet 20 cau hoi |

Console se in ra:
- Tra loi tung cau hoi cua ca 2 phuong phap
- Tong hop token, thoi gian
- Chenh lech chi phi giua 2 phuong phap

## 20 cau hoi benchmark

| # | Cau hoi | Loai |
|---|---------|------|
| 1 | OpenAI duoc thanh lap nam nao? | Single-hop |
| 2 | Ai la nguoi sang lap Google? | Single-hop |
| ... | ... | ... |
| 11 | CEO cua cong ty duoc Elon Musk dong sang lap la ai? | Multi-hop |
| 12 | Microsoft da mua lai cong ty nao trong linh vuc mang xa hoi? | Multi-hop |
| ... | ... | ... |

## Luu y

- **Chi phi API**: Chay full benchmark 20 cau hoi ton khoang 20-50k tokens (~$0.01-0.05 voi gpt-4o-mini)
- **Thoi gian**: Khoang 2-5 phuy tuy vao toc do API
- Co the giam so cau hoi benchmark bang cach sua danh sach `BENCHMARK_QUESTIONS` trong `main.py`
