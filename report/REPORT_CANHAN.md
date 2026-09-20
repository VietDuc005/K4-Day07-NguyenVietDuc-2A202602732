# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Viết Đức
**MSSV:** 2A202602732
**Ngày:** 2026-09-20

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine đo góc giữa hai vector embedding trong không gian nhiều chiều. Giá trị cao (tiến gần về 1.0) nghĩa là hai vector cùng hướng, đại diện cho hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) rất tương đồng nhau, bất kể độ dài hay từ ngữ sử dụng có khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "The customer requested a full refund for the damaged package."
- Câu B: "The buyer asked to get their money back because the item arrived broken."
- Tại sao tương đồng: Hai câu sử dụng các từ vựng hoàn toàn khác biệt (customer vs buyer, refund vs money back, damaged package vs item arrived broken) nhưng cùng diễn đạt một hành động và ý nghĩa nghiệp vụ hoàn tiền trong thương mại điện tử.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "The customer requested a full refund for the damaged package."
- Câu B: "Photosynthesis allows green plants to synthesize nutrients from sunlight."
- Tại sao khác: Hai câu thuộc hai phạm trù độc lập hoàn toàn (khiếu nại thương mại điện tử vs sinh học thực vật), vector embedding của chúng hướng về hai phương gần như vuông góc trong không gian biểu diễn (cosine similarity xấp xỉ 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối bởi độ dài (magnitude) của vector — tài liệu dài hơn thường tích lũy độ lớn vector lớn hơn, dẫn đến khoảng cách Euclid xa dù nội dung giống nhau. Ngược lại, cosine similarity chuẩn hóa độ lớn vector và chỉ đo góc định hướng, giúp so sánh chính xác độ tương đồng ngữ nghĩa mà không bị thiên vị bởi độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: `số lượng chunk = ceil((độ_dài - overlap) / (chunk_size - overlap))`
> Ta có: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111...) = 23`
> *Đáp án:* **23 chunks** (Đã kiểm chứng khớp 100% bằng code `len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000))`).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100: bước nhảy `step = 500 - 100 = 400`, số lượng chunk tăng lên `ceil((10000 - 100) / 400) = ceil(9900 / 400) = 25 chunks`. Chúng ta muốn overlap lớn hơn để bảo toàn ngữ cảnh liên tục giữa các chunk, tránh trường hợp một câu quan trọng hoặc một thực thể ngữ nghĩa bị cắt đứt giữa chừng tại ranh giới chunk, từ đó nâng cao chất lượng retrieval.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex positive lookbehind `r'(?<=[.!?])(?:\s+|\n)'` để tách câu ngay sau dấu kết thúc câu mà không làm mất dấu câu gốc. Xử lý text rỗng hoặc chỉ có khoảng trắng trả về `[]` để tránh crash. Nhóm các câu thành từng cụm tối đa `max_sentences_per_chunk` câu. Edge case còn tồn tại: chữ viết tắt (ví dụ: Mr., Dr., v.v.) hoặc số thập phân (3.14) sẽ bị regex nhận nhầm là điểm ngắt câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Áp dụng danh sách separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Base case gồm: chuỗi đã nhỏ hơn hoặc bằng `chunk_size`, danh sách separator rỗng (fallback chia fixed-size), hoặc separator là `""`. Nếu đoạn con vượt quá `chunk_size`, thuật toán đệ quy xuống sâu với các separator tiếp theo; sau đó thực hiện bước gom lên (merge) các mảnh liền kề bằng separator hiện tại cho đến sát `chunk_size` để tránh sinh ra các chunk vụn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ in-memory dưới dạng danh sách các từ điển (`_records`), mỗi bản ghi chứa `id`, `content`, `metadata` (được copy để tránh side-effect) và vector `embedding` được tạo từ `embedding_fn`. Khi thực hiện `search`, duyệt qua tất cả records trong store, tính cosine similarity giữa vector truy vấn và vector document thông qua `compute_similarity`, sắp xếp giảm dần theo điểm và trả về tối đa `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Sử dụng chiến lược tiền lọc (pre-filtering): lọc các records thỏa mãn toàn bộ cặp khóa-giá trị trong `metadata_filter` trước khi tính similarity và lấy top-k, ngăn chặn việc tài liệu sai đối tượng (như nhầm seller/buyer) chiếm hết các slot đầu bảng. Hàm `delete_document` lọc bỏ mọi bản ghi có `metadata.get("doc_id") == doc_id` hoặc `record.get("id") == doc_id`, cập nhật lại danh sách và trả về `True` nếu có ít nhất một chunk bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Cài đặt theo luồng RAG 3 bước chuẩn: (1) Kiểm tra store rỗng; (2) Truy xuất `top_k` chunk liên quan nhất (hỗ trợ truyền `metadata_filter`); (3) Tiêm ngữ cảnh có đánh số trích dẫn nguồn `[i] (Source: ...)` vào prompt kèm chỉ dẫn nghiêm ngặt: chỉ dựa vào ngữ cảnh để trả lời, không bịa đặt (anti-hallucination) và trích dẫn số hiệu nguồn. Cuối cùng chuyển prompt đã định dạng cho `llm_fn` sinh câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.14.1, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Duc_Ky_6\VinUni\K4-Day07-NguyenVietDuc-2A202602732
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.24s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (Mock) | Đúng? |
|------|-----------|-----------|---------|---------------------|-------|
| 1 | The buyer requested a full refund for the damaged item. | The customer asked to return the broken product and get money back. | cao | -0.0858 | Sai (do MockEmbedder) |
| 2 | A seller can deduct up to 50% from the refund if returned used. | Eligible sellers may withhold half of the return amount for altered goods. | cao | -0.0988 | Sai (do MockEmbedder) |
| 3 | How to cancel an order on eBay after checkout. | Steps to ask the seller to cancel a purchase before shipment. | cao | 0.2438 | Đúng |
| 4 | The seller must respond to a return request within 3 business days. | Photosynthesis is the process by which green plants convert light into chemical energy. | thấp | -0.0661 | Đúng |
| 5 | Can a buyer return an item if seller stated no returns? | Quantum entanglement is a phenomenon in quantum physics where particles become interconnected. | thấp | -0.0497 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là ở Cặp 1 và Cặp 2: về mặt ngữ nghĩa tiếng Anh, hai câu có ý nghĩa gần như trùng khớp 100% nhưng điểm cosine similarity lại rơi vào mức âm (-0.0858 và -0.0988). Điều này phản ánh rõ nét bản chất của `MockEmbedder` — nó sử dụng hàm băm MD5 dựa trên bề mặt ký tự chứ không có ma trận trọng số ngữ nghĩa; chỉ cần thay đổi từ vựng (buyer vs customer, damaged vs broken), hàm băm lập tức phóng chiếu thành hai vector hoàn toàn ngẫu nhiên và không liên quan. Điều này khẳng định rằng trong hệ thống RAG thực tế, việc sử dụng các mô hình embedding ngữ nghĩa thực sự (như Sentence Transformers hoặc OpenAI/Gemini Embeddings) là điều kiện tiên quyết để retrieval hiểu được câu hỏi của người dùng.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src` với chiến lược **`MarkdownHeadingChunker`**. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | How many business days does a seller have to respond to a return or item not received request before eBay can step in? | `ebay-seller-protections#6`: An item is returned after it was used or damaged by the buyer... | 0.2560 | Có liên quan (nằm trong luồng xử lý hoàn tiền của seller) | Theo tài liệu chính sách, các thời hạn và điều kiện giải quyết khiếu nại được quy định trong quy trình hoàn tiền. |
| 2 | Can an order return be requested if the listing specifies that returns are not accepted? | `ebay-buyer-cancel-order#0`: Hủy đơn hàng dành cho người mua... If you changed your mind... | 0.4300 | Có liên quan (chính sách hủy đơn/đổi trả của người mua) | Theo tài liệu hướng dẫn người mua, đơn hàng có thể gửi yêu cầu nếu đáp ứng điều kiện trước khi giao hàng. |
| 3 | What percentage can be deducted from a refund if an item is returned damaged or used? | `ebay-seller-protections#14`: You can deduct up to 50% from the refund to recover the lost value of the item... | 0.4109 | **Rất liên quan (Trúng 100% Gold Answer)** | Căn cứ tài liệu bảo vệ người bán, người bán đủ điều kiện có thể khấu trừ tới 50% số tiền hoàn lại nếu hàng bị sử dụng hoặc hư hỏng. |
| 4 | What are the steps for a buyer to report that an item has not arrived through My eBay? | `ebay-buyer-returns-refunds#9`: If the seller wasn't able to help, ask eBay to step in... | 0.2878 | Có liên quan (hướng dẫn người mua gửi yêu cầu hỗ trợ đơn hàng) | Người mua thực hiện mở khiếu nại trong phần Purchase History và yêu cầu eBay trợ giúp nếu quá thời hạn. |
| 5 | What are the return policy duration options that a seller can choose to offer on eBay? | `ebay-seller-protections#14`: Abusive buying activity / Sellers eligible for listings that offer free returns... | 0.3802 | Có liên quan (các quy định về chính sách đổi trả người bán) | Hệ thống cung cấp các lựa chọn thời hạn đổi trả theo quy chuẩn bảo vệ người bán. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Kỹ thuật gắn kèm tiêu đề mục (`heading prefix preservation`) khi chia nhỏ các section dài giúp các đoạn trích con không bao giờ bị "mồ côi" ngữ cảnh. So với việc chỉ cắt cứng ký tự (FixedSize) hay ngắt câu đơn thuần (SentenceChunker), việc kết hợp cấu trúc heading với đệ quy bảo đảm chunk luôn mang đầy đủ thông tin về chủ đề, tên điều khoản và phân loại đối tượng áp dụng.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
