# Multi-Agent Research System: Benchmark Report

Đặng Văn Nhân - 2A202601050
Link trace:

## 1. Metrics Comparison

| Run                         | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes                                                                                  |
| --------------------------- | ----------: | ---------: | ------: | ------------: | -----------: | -------------------------------------------------------------------------------------- |
| **Single-Agent**            |       14.03 |    $0.0005 |  8.0/10 |            0% |           0% | 1 prompt trực tiếp, 0 nguồn tham khảo ngoài, không có trích dẫn                        |
| **Multi-Agent (LangGraph)** |       37.46 |    $0.0025 | 10.0/10 |          100% |           0% | 5 bước (researcher → analyst → writer → critic → done), 5 nguồn thực tế, 100% citation |

---

## 2. Phân tích chi tiết kết quả thực nghiệm

### Single-Agent Baseline

- **Thời gian (Latency)**: 14.03s (nhanh hơn ~2.6 lần).
- **Chi phí (Cost)**: $0.0005 (thấp hơn ~5 lần).
- **Đặc điểm**: Trả lời tổng quan khá tốt về mặt cấu trúc chung nhưng chỉ dựa trên tri thức có sẵn trong trọng số mô hình (parametric knowledge). Không có nguồn tài liệu thực tế, **độ phủ trích dẫn đạt 0%**, tiềm ẩn nguy cơ ảo giác (hallucination) khi gặp các câu hỏi chuyên sâu hoặc dữ liệu mới.

### Multi-Agent System (LangGraph)

- **Thời gian (Latency)**: 37.46s (do trải qua tuần tự 5 node: Supervisor → Researcher → Analyst → Writer → Critic).
- **Chi phí (Cost)**: $0.0025.
- **Đặc điểm**:
  - **Researcher**: Gọi Tavily API lấy 5 nguồn tài liệu uy tín về RAG và Fine-tuning.
  - **Analyst**: Phân tích đối chiếu chuyên sâu về cơ chế, trade-off, độ ổn định và khoảng trống kiến thức.
  - **Writer**: Tổng hợp báo cáo học thuật hoàn chỉnh kèm đánh số trích dẫn `[1]` đến `[5]` liên kết trực tiếp với mục `### References`.
  - **Critic**: Độc lập thẩm định đạt chuẩn **PASS**, xác nhận 0% hallucination và 100% citation grounding.

---

## 3. Failure Modes gặp phải và Cách khắc phục (Failure Modes & Mitigations)

1. **Failure Mode 1: Infinite Routing Loop / Lặp vô hạn đốt Token**
   - _Nguyên nhân_: Supervisor có thể bị kẹt trong vòng lặp chuyển giao giữa các agent nếu điều kiện kết thúc không rõ ràng hoặc một agent liên tục trả về kết quả chưa đạt yêu cầu.
   - _Cách fix_: Cài đặt **Guardrail** cứng với `max_iterations = 6` trong `SupervisorAgent` và điều kiện chuyển hướng an toàn về `done` khi đạt ngưỡng lặp tối đa.

2. **Failure Mode 2: Cascading Hallucinations & Thiếu trích dẫn nguồn**
   - _Nguyên nhân_: Writer có thể sinh thêm các số liệu hoặc kết luận không có trong tài liệu nghiên cứu ban đầu mà không có cơ chế rà soát.
   - _Cách fix_: Tích hợp `CriticAgent` độc lập vào LangGraph workflow ngay sau Writer để thực hiện Fact-Checking, đối soát từng câu trích dẫn với `state.sources` trước khi chốt báo cáo cuối cùng.

3. **Failure Mode 3: Lỗi SSL Certificate trên macOS khi gọi HTTPS**
   - _Nguyên nhân_: Python trên macOS không tự động liên kết với hệ thống chứng chỉ CA của OS.
   - _Cách fix_: Sử dụng bundle chứng chỉ từ thư viện `certifi` (`certifi.where()`) và cấu hình `httpx.Client` với timeout và retry policy qua `tenacity`.

---

## 4. Exit Ticket: Khi nào nên / không nên dùng Multi-Agent?

### Câu 1: Case nào NÊN dùng Multi-Agent? Vì sao?

- **Trường hợp áp dụng**:
  1. Các tác vụ nghiên cứu chuyên sâu (Deep Research), phân tích thị trường, tổng hợp tài liệu khoa học đòi hỏi tìm kiếm đa nguồn và xác minh tính đúng đắn.
  2. Các bài toán phức tạp đòi hỏi sự phân hóa vai trò rõ ràng (Domain Experts: Lập trình viên, Reviewer, Tester; hoặc Thu thập dữ liệu, Phân tích, Viết bài, Thẩm định).
  3. Hệ thống yêu cầu độ tin cậy và khả năng kiểm toán (Auditability/Observability) cao, cần biết chính xác agent nào đưa ra thông tin nào.
- **Lý do**:
  - Phân tách bài toán lớn thành các context nhỏ chuyên biệt giúp LLM không bị **loãng context (context drift)**.
  - Cho phép chèn các bước kiểm soát chất lượng độc lập (như Critic/Verifier) giữa các khâu.

### Câu 2: Case nào KHÔNG NÊN dùng Multi-Agent? Vì sao?

- **Trường hợp áp dụng**:
  1. Các tác vụ đơn giản, truy vấn thông thường (hỏi đáp định nghĩa, dịch thuật ngắn, tóm tắt 1 đoạn văn bản).
  2. Các ứng dụng tương tác thời gian thực yêu cầu độ trễ cực thấp (Low Latency < 2 giây như trợ lý đàm thoại giọng nói, chatbot CSKH phản hồi tức thì).
  3. Dự án có ngân sách chi phí token bị giới hạn nghiêm ngặt.
- **Lý do**:
  - Hệ thống Multi-Agent có độ trễ lớn hơn nhiều lần do phải thực hiện nhiều lời gọi API nối tiếp nhau.
  - Chi phí token tăng gấp 3 - 5 lần và độ phức tạp bảo trì/vận hành workflow tăng đáng kể mà không mang lại giá trị tương xứng cho các tác vụ đơn giản.
