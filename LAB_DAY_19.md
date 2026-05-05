# LAB DAY 19: XÂY DỰNG HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS

## 1. MỤC TIÊU BÀI HỌC
- Hiểu rõ quy trình trích xuất thực thể (Entity Extraction) và quan hệ (Relation Extraction) từ văn bản thô.
- Làm quen với các thư viện quản lý đồ thị: NetworkX, Neo4j và framework mã nguồn mở NodeRAG.
- Xây dựng hoàn chỉnh một pipeline GraphRAG: từ lập chỉ mục (Indexing) đến truy vấn đa bước (Multi-hop Querying).
- Đánh giá sự khác biệt về độ chính xác giữa Flat RAG và GraphRAG.

## 2. PHẦN 1: NGHIÊN CỨU VÀ CHUẨN BỊ (RESEARCH)

### 2.1. Quy trình xử lý dữ liệu đồ thị
1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
3. Query Answering: Sự khác biệt giữa BFS và tìm kiếm vector là gì?

### 2.2. Tìm hiểu công cụ
- **NetworkX**: Thư viện Python nghiên cứu mạng lưới.
- **Neo4j**: CSDL đồ thị, dùng Cypher.
- **NodeRAG**: Framework GraphRAG trên NetworkX.

## 3. PHẦN 2: ENVIRONMENT SETUP

```bash
pip install networkx matplotlib neo4j openai pandas
pip install noderag
pip install langchain langchain-openai
```

## 4. PHẦN 3: HƯỚNG DẪN

### Bước 1: Indexing
Input:
> "OpenAI được thành lập bởi Sam Altman và Elon Musk vào năm 2015."

Output:
- (OpenAI, FOUNDED_BY, Sam Altman)
- (OpenAI, FOUNDED_BY, Elon Musk)
- (OpenAI, FOUNDED_IN, 2015)

### Bước 2: Construction
- NetworkX
- Neo4j
- NodeRAG

### Bước 3: Querying
1. Nhận câu hỏi
2. Extract entity
3. Traverse 2-hop
4. Textualization + LLM

### Bước 4: Evaluation
- So sánh Flat RAG vs GraphRAG
- Ghi lại hallucination

## 5. RECOMMENDATIONS

| Mục tiêu | Tool | Lý do |
|----------|------|------|
| Dễ bắt đầu | NodeRAG | Tích hợp sẵn |
| Trực quan | Neo4j | GUI mạnh |
| Nghiên cứu | NetworkX | Linh hoạt |

## 6. DELIVERABLES
1. Code (.py/.ipynb)
2. Screenshot graph
3. Benchmark 20 câu hỏi
4. Phân tích chi phí
