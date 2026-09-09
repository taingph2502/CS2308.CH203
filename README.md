# CS2308.CH203

## KẾ HOẠCH

### 1. Finetune
- Base model: Qwen/Qwen2.5-3B-Instruct
- Kỹ thuật: PiSSA (có trong thư viện PEFT)
- Dataset: hungnm/vietnamese-medical-qa (dùng 3-5k mẫu)

### 2. RAG
- Corpus: tarudesu/ViHealthQA
- Retrieval: BAAI/bge-m3 + BM25 + RRF
- Rerank: BAAI/bge-reranker-v2-m3
- Model: Qwen/Qwen2.5-3B-Instruct (đã finetune)

### 3. Ablation Study
- Metric: 
  - Retrieval: MRR
  - Generation: PhoBERT-BERTScore, BGE-M3 Cosine Similarity
- Đối tượng đánh giá generation:
  - qwen2.5
  - qwen2.5 + RAG
  - qwen2.5 + PiSSA
  - qwen2.5 + PiSSA + RAG
 
## LƯU Ý

### 1. Finetune

1. **Cách lưu và merge trọng số PiSSA trong PEFT:**
* Khác với LoRA thông thường (khởi tạo adapter bằng 0), PiSSA phân rã ma trận gốc qua SVD, nên trọng số gốc của base model sẽ bị sửa đổi để khớp với phần dư.
* *Lưu ý:* Sau khi train xong, **hãy dùng `model.merge_and_unload()` rồi lưu full model ra một thư mục mới**. Tránh việc chỉ lưu mỗi folder adapter rồi sau này load đè lên base model gốc theo cách thông thường, rất dễ bị lệch trọng số nếu cấu hình không khớp.

2. **Masking Loss (Chỉ tính loss trên câu trả lời của bác sĩ):**
* Đừng để mô hình tính loss trên cả phần câu hỏi của người dùng. Hãy dùng `DataCollatorForCompletionOnlyLM` (của thư viện `trl`) hoặc gán nhãn `-100` cho toàn bộ các token thuộc lượt hỏi của User. Mô hình chỉ cần học cách *sinh câu trả lời*, không cần học lại câu hỏi.

3. **Tuân thủ đúng ChatML Template:**
* Qwen2.5 dùng cú pháp ChatML (`<|im_start|>user...`). Hãy luôn nạp dữ liệu qua hàm `tokenizer.apply_chat_template(..., tokenize=False)` thay vì tự nối chuỗi bằng tay bằng dấu `\n`.

4. **Độ dài chuỗi (`max_seq_length`):**
* Giới hạn `max_seq_length` khoảng **512 – 768 tokens**. Bộ dữ liệu y tế hỏi đáp thường ngắn, đặt 2048 hoặc 4096 tokens sẽ gây tốn VRAM không cần thiết và làm chậm tốc độ huấn luyện.

### 2. RAG

1. **BM25 tiếng Việt bắt buộc phải tách từ (*Word Segmentation*):**
* BM25 mặc định chia từ theo khoảng trắng. Trong tiếng Việt, nếu không tách từ ghép, từ `"bệnh nhân"` sẽ bị hiểu thành hai từ rời rạc là `"bệnh"` và `"nhân"`.
* *Giải pháp:* Dùng `pyvi` hoặc `underthesea` để chuẩn hóa văn bản trước khi đưa vào BM25: `"bệnh nhân"` $\rightarrow$ `"bệnh_nhân"`, `"huyết áp"` $\rightarrow$ `"huyết_áp"`.

2. **Gán `doc_id` tường minh cho từng Chunk trong DB:**
* Để tính được chỉ số MRR ở Phần 3, bạn **bắt buộc phải biết chính xác đoạn văn nào là đáp án gốc**.
* Khi nạp câu trả lời của tập `test` vào DB, hãy gán thêm metadata: ví dụ `metadata={"doc_id": "test_01"}`. Khi truy vấn câu hỏi số 01, bạn chỉ cần kiểm tra xem trong danh sách trả về, chunk có `doc_id == "test_01"` nằm ở vị trí thứ mấy. Nếu so khớp bằng text thông thường, chỉ cần một dấu cách hay ngắt dòng khác nhau là so sánh chuỗi sẽ bị sai.

3. **Phân bổ số lượng Top-K qua từng tầng:**
* Đừng đưa quá nhiều văn bản vào Reranker vì Cross-Encoder tính toán rất nặng:
* *BM25:* Lấy top 30.
* *BGE-M3 Dense:* Lấy top 30.
* *RRF ($k=60$):* Gộp lại và lấy top 15 văn bản điểm cao nhất.
* *BGE-Reranker-v2-m3:* Chấm điểm lại 15 văn bản này và chỉ chọn lấy **Top 3** để đưa vào Prompt cho Qwen.

### 3. Ablation Study

1. **Tắt tính ngẫu nhiên khi sinh câu trả lời (Greedy Decoding):**
* Khi so sánh giữa nhánh *(1) Chỉ có PiSSA* và nhánh *(2) PiSSA + RAG*, bắt buộc phải cấu hình:
```python
do_sample=False, temperature=0.0
```

* Nếu bật ngẫu nhiên (nhiệt độ > 0), cùng một câu hỏi chạy 2 lần sẽ ra câu trả lời khác nhau, làm kết quả so sánh PhoBERT-Score bị dao động do yếu tố may rủi chứ không phản ánh đúng tác động của RAG.

2. **Kỹ thuật "Khóa đuôi" (Prompt Anchoring) cho nhánh RAG:**
* Vì mô hình được finetune trên tập QA trực tiếp (không có ngữ cảnh RAG), ở nhánh RAG, hãy đặt câu hướng dẫn ngay sát câu hỏi:
```text
Tài liệu: {context}
Câu hỏi: {question}
Lưu ý quan trọng: Chỉ sử dụng thông tin trong tài liệu trên để trả lời.
```

3. **Cài đặt đúng tokenizer cho PhoBERT-BERTScore:**
* Khi gọi thư viện `bert_score`, hãy truyền thêm tham số tách từ cho tiếng Việt để BERTScore tính điểm dựa trên token từ vựng chuẩn xác:
```python
score(cands, refs, model_type="vinai/phobert-base", lang="vi")
```

* **Bước A (Chạy Retrieval):** Nạp BGE-M3 + BM25 + Reranker $\rightarrow$ Truy vấn 50 câu test $\rightarrow$ Lưu kết quả Top-3 context và điểm MRR ra file `rag_results.jsonl`. Sau đó giải phóng GPU (`del model; torch.cuda.empty_cache()`).
* **Bước B (Chạy Sinh văn bản):** Nạp model Qwen2.5-3B (PiSSA) $\rightarrow$ Đọc `rag_results.jsonl` $\rightarrow$ Sinh câu trả lời cho cả 2 nhánh (có RAG và không RAG) $\rightarrow$ Lưu ra file `generated_answers.jsonl`. Giải phóng GPU.
* **Bước C (Chạy Chấm điểm):** Nạp PhoBERT & BGE-M3 $\rightarrow$ Đọc `generated_answers.jsonl` và chấm điểm BERTScore + Cosine Sim $\rightarrow$ Xuất bảng kết quả cuối cùng.
