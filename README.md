# CS2308.CH203

## Retrieval Ablation

Bảng này đo lường xem các thuật toán tìm kiếm tài liệu có đưa được **tài liệu y khoa chính xác (Gold Document)** lên các vị trí đầu tiên hay không (trên 1.000 câu hỏi test):

| Method | MRR | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.3870 | 0.3110 | 0.4110 | 0.4680 | 0.5410 |
| bge_m3 | 0.5388 | 0.4480 | 0.5910 | 0.6380 | 0.7040 |
| hybrid_rrf | 0.4827 | 0.3850 | 0.5270 | 0.5890 | 0.6670 |
| hybrid_reranker | **0.5755** | **0.4910** | **0.6190** | **0.6770** | **0.7400** |

**Các chỉ số cột:**
- **MRR** (Mean Reciprocal Rank - Điểm xếp hạng nghịch đảo): Thước đo vị trí trung bình của tài liệu đúng. Nếu tài liệu đúng nằm ở Top 1 $\rightarrow$ điểm là $1/1 = 1.0$; nằm ở Top 2 $\rightarrow$ điểm là $1/2 = 0.5$; Top 3 $\rightarrow$ $1/3 = 0.33$. Điểm càng gần 1.0 càng tốt.
- **Hit@K** (Tỷ lệ trúng Top-K): Tỷ lệ % các câu hỏi mà tài liệu đúng xuất hiện trong Top K kết quả trả về.

**So sánh các phương pháp hàng:**
- **bm25** (Tìm kiếm từ khóa): Kém nhất (MRR 0.3870) vì không hiểu được từ đồng nghĩa y khoa (ví dụ: "đau đầu" vs "nhức nửa đầu").
- **bge_m3** (Dense Embedding): Vượt trội hơn hẳn (MRR 0.5388, Hit@10 đạt 70.4%).
- **hybrid_reranker** (Kết hợp BM25 + BGE-M3 rồi dùng Re-ranker sắp xếp lại Top 100): Đạt điểm cao nhất toàn diện (MRR 0.5755, Hit@10 đạt 74%). Đây là lý do phương pháp này được chọn làm đầu vào để cấp ngữ cảnh cho nhánh RAG.

## Generation Ablation

| System | PhoBERT-BERTScore F1 (mean [CI95]) | BGE-M3 cosine (mean [CI95]) |
|---|---:|---:|
| qwen | 0.6237 [0.6212, 0.6261] | 0.7817 [0.7768, 0.7866] |
| qwen_rag | **0.6764** [0.6721, 0.6806] | **0.8134** [0.8077, 0.8191] |
| qwen_pissa | 0.5975 [0.5938, 0.6012] | 0.7505 [0.7449, 0.7561] |
| qwen_pissa_rag | *0.6643* [0.6569, 0.6717] | *0.7881* [0.7813, 0.7947] |

**Ý nghĩa của 2 thước đo:**

- **PhoBERT-BERTScore F1:**
Đo độ tương đồng ngữ nghĩa từng từ (token-level) giữa câu trả lời mô hình sinh ra và câu trả lời chuẩn của bác sĩ.
Sử dụng mô hình ngôn ngữ tiếng Việt chuyên sâu PhoBERT (sau khi đã tách từ chuẩn qua VnCoreNLP). Thang điểm 0 – 1, càng cao nghĩa là câu trả lời càng bám sát ý tứ ngữ nghĩa của câu trả lời mẫu.

- **BGE-M3 cosine:**
Đo độ tương đồng ngữ nghĩa toàn bộ văn bản (Sentence-level Semantic Cosine Similarity). Đo xem tổng thể câu trả lời có cùng chủ đề, định hướng chẩn đoán với câu mẫu hay không.

- **Ý nghĩa của [CI95] (Khoảng tin cậy Bootstrap 95%):**
Ví dụ: 0.6764 [0.6721, 0.6806] nghĩa là điểm trung bình là 0.6764, và ta có 95% độ tin cậy rằng điểm thực tế nằm trong dải hẹp từ 0.6721 đến 0.6806.
Khoảng tin cậy của qwen_rag hoàn toàn không bị đè (overlap) lên các khoảng khác $\rightarrow$ Sự vượt trội của RAG là có ý nghĩa thống kê thực sự chứ không phải ngẫu nhiên.
