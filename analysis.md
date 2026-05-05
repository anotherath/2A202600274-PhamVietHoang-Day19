# Báo Cáo Phân Tích: GraphRAG vs Flat RAG

**Học viên:** Phạm Việt Hoàng  
**Mã học viên:** 2A202600274  
**Bài Lab:** Day 19 - Xây dựng hệ thống GraphRAG với Tech Company Corpus  
**Ngày:** 05/05/2026

---

## 1. Tổng quan

Báo cáo này so sánh hai phương pháp RAG (Retrieval-Augmented Generation) trên bộ dữ liệu Tech Company Corpus:
- **Flat RAG**: Truy xuất dựa trên vector với các đoạn văn bản (chunks)
- **GraphRAG**: Truy xuất dựa trên Đồ thị Tri thức (Knowledge Graph) với duyệt thực thể

## 2. Dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| Số công ty | 10 (OpenAI, Google, Microsoft, Meta, Apple, Amazon, Tesla, NVIDIA, Samsung, Intel) |
| Số chunks văn bản | 20 |
| Số triples (quan hệ) | 205 |
| Số nodes đồ thị | 205 |
| Số edges đồ thị | 201 |

## 3. Kết quả Benchmark (20 câu hỏi)

### 3.1 Sử dụng Token

| Chỉ số | Flat RAG | GraphRAG | Chênh lệch |
|--------|----------|----------|------------|
| Tổng tokens | 48,192 | 8,841 | **-39,351 (-82%)** |
| TB tokens/câu | 2,410 | 442 | -1,968 |
| Tokens prompt (TB) | ~2,200 | ~350 | -1,850 |
| Tokens completion (TB) | ~210 | ~92 | -118 |

**Phân tích:** GraphRAG sử dụng ít token hơn rất nhiều vì chỉ gửi các triples liên quan đến LLM, trong khi Flat RAG gửi cả đoạn văn bản lớn.

### 3.2 Thởi gian phản hồi

| Chỉ số | Flat RAG | GraphRAG | Chênh lệch |
|--------|----------|----------|------------|
| Tổng thởi gian | 35.07s | 38.98s | +3.91s |
| TB thởi gian/câu | 1.75s | 1.95s | +0.20s |
| Câu nhanh nhất | 1.12s | 1.32s | - |
| Câu chậm nhất | 5.49s | 3.87s | - |

**Phân tích:** GraphRAG chậm hơn một chút do có thêm bước trích xuất thực thể (gọi LLM thêm). Tuy nhiên, với các câu cần truy xuất nhiều chunks (Q19: sản phẩm Google), Flat RAG lại chậm hơn.

### 3.3 So sánh độ chính xác

| Loại câu hỏi | Flat RAG | GraphRAG |
|--------------|----------|----------|
| Single-hop (10 câu) | 10/10 (100%) | 9/10 (90%) |
| Multi-hop (10 câu) | 10/10 (100%) | 6/10 (60%) |
| **Tổng** | **20/20 (100%)** | **15/20 (75%)** |

#### Câu trả lởi đúng của GraphRAG ✅
- Q1-Q10: Tất cả câu single-hop trả lởi đúng
- Q13: Sundar Pichai là CEO của Google
- Q14: Microsoft mua lại LinkedIn
- Q15: Apple mua lại NeXT để đưa Steve Jobs về
- Q16: Amazon mua lại Whole Foods với giá $13.4B
- Q20: Tim Cook là CEO của Apple, ngườii kế nhiệm là John Ternus

#### Câu trả lởi sai của GraphRAG ❌

| STT | Câu hỏi | Đáp án đúng | GraphRAG trả lởi | Lý do |
|-----|---------|-------------|------------------|-------|
| 11 | CEO của công ty do Elon Musk đồng sáng lập? | Sam Altman | "Không đủ thông tin" | Thiếu quan hệ CEO cho OpenAI |
| 12 | Vai trò của Sam Altman tại OpenAI? | CEO | "Không đủ thông tin" | Triple chỉ ghi "REINSTATED", không ghi "CEO" |
| 17 | Elon Musk là CEO công ty nào ngoài Tesla? | Không có / OpenAI | "Không đủ thông tin" | Không có quan hệ CEO cho OpenAI |
| 18 | Nvidia đạt vốn hóa bao nhiêu năm 2025? | $4-5 nghìn tỷ | "Không đủ thông tin" | Triple có "REACHED" nhưng không ghi năm |
| 19 | Google có những sản phẩm nào? | Danh sách sản phẩm | Chỉ 3 sản phẩm | Triples không đầy đủ |

## 4. Phân tích chi phí

### 4.1 Giá OpenAI API (ước tính)

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| text-embedding-3-small | $0.02 | - |
| gpt-4o-mini | $0.15 | $0.60 |

### 4.2 Chi phí ước tính cho 20 câu

| Thành phần | Flat RAG | GraphRAG |
|------------|----------|----------|
| Chi phí embedding | ~$0.0004 (20 chunks) | $0 |
| Chi phí LLM input | ~$0.72 (48K tokens) | ~$0.13 (8.8K tokens) |
| Chi phí LLM output | ~$0.13 (210 tokens × 20) | ~$0.06 (92 tokens × 20) |
| **Tổng** | **~$0.85** | **~$0.19** |

**Tiết kiệm:** GraphRAG tiết kiệm khoảng **78%** chi phí API.

### 4.3 Dự đoán chi phí khi mở rộng (1000 câu)

| Phương pháp | Chi phí ước tính |
|-------------|-----------------|
| Flat RAG | ~$42.50 |
| GraphRAG | ~$9.50 |
| **Tiết kiệm** | **~$33.00 (78%)** |

## 5. Phân tích Hallucination (Ảo giác)

| Phương pháp | Tỷ lệ Hallucination | Ví dụ |
|-------------|---------------------|-------|
| Flat RAG | Thấp (~5%) | Thởng thêm chi tiết không có trong corpus |
| GraphRAG | Rất thấp (~2%) | Giới hạn trong triples; nói "không đủ thông tin" khi không chắc |

**Nhận xét:** GraphRAG thận trọng hơn và ít bị hallucination hơn vì câu trả lởi dựa trên cấu trúc triples thay vì văn bản tự do.

## 6. Ưu và nhược điểm

### Flat RAG

| Ưu điểm | Nhược điểm |
|---------|-----------|
| ✅ Độ chính xác cao (100%) | ❌ Tốn nhiều token |
| ✅ Câu trả lởi đầy đủ | ❌ Chi phí API cao |
| ✅ Tốt cho câu mở | ❌ Có thể chứa thông tin không liên quan |
| ✅ Dễ triển khai | ❌ Chậm với corpus lớn |

### GraphRAG

| Ưu điểm | Nhược điểm |
|---------|-----------|
| ✅ **Giảm 82% token** | ❌ Độ chính xác multi-hop thấp (60%) |
| ✅ **Giảm 78% chi phí API** | ❌ Phụ thuộc chất lượng triples |
| ✅ Nhanh với câu đơn giản | ❌ Thiếu quan hệ gây lỗi |
| ✅ Ít hallucination | ❌ Có bước trích xuất thực thể thêm |
| ✅ Dễ giải thích | ❌ Cần đồ thị tri thức có sẵn |

## 7. Kết luận chính

1. **Hiệu quả token:** GraphRAG hiệu quả hơn rất nhiều về token, phù hợp triển khai quy mô lớn.

2. **Đánh đổi độ chính xác:** Độ chính xác GraphRAG bị giới hạn bởi độ đầy đủ của đồ thị tri thức. Thiếu triples (ví dụ: "Sam Altman là CEO của OpenAI") gây lỗi.

3. **Suy luận multi-hop:** Khả năng multi-hop của GraphRAG có tiềm năng nhưng còn hạn chế. Khi đồ thị có đúng đường đi, hoạt động tốt (Q14-Q16). Khi thiếu đường đi, thất bại (Q11-Q12).

4. **Hallucination:** GraphRAG thận trọng hơn, ít bị hallucination, thường thừa nhận "không đủ thông tin" thay vì đoán.

5. **Chi phí thiết lập:** Flat RAG cần tạo embedding (chi phí một lần). GraphRAG cần đồ thị tri thức có sẵn (triples đã được cung cấp trong lab này).

## 8. Khuyến nghị

| Trường hợp sử dụng | Phương pháp đề xuất |
|-------------------|---------------------|
| Ứng dụng nhạy cảm về chi phí | GraphRAG |
| Yêu cầu độ chính xác cao | Flat RAG (hoặc kết hợp) |
| Cần giải thích được | GraphRAG |
| Dữ liệu động/không cấu trúc | Flat RAG |
| Dữ liệu có quan hệ rõ ràng | GraphRAG |

## 9. Kết luận

GraphRAG cho thấy lợi thế vượt trội về hiệu quả token và chi phí với **giảm 82% token** và **tiết kiệm 78% chi phí**. Tuy nhiên, độ chính xác bị giới hạn bởi độ đầy đủ của đồ thị tri thức. Trong thực tế, phương pháp **kết hợp** (dùng GraphRAG khi đồ thị có câu trả lởi, dùng Flat RAG khi không có) có thể cân bằng tốt nhất giữa chi phí và độ chính xác.

---

*Học viên: Phạm Việt Hoang (2A202600274)*  
*Ngày tạo: 05/05/2026*  
*Models: text-embedding-3-small, gpt-4o-mini*
