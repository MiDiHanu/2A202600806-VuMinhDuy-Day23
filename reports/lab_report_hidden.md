# Day 08 Lab Report — Xây dựng Agent hỗ trợ Ticket với LangGraph

## 1. Thông tin sinh viên

- Họ tên: Vũ Minh Duy
- MSSV: 2A202600806
- Nhà cung cấp LLM: Google Gemini (cloud) qua thư viện `langchain-google-genai`

## 2. Thiết kế hệ thống

Tôi sử dụng `StateGraph` với một `AgentState` có kiểu rõ ràng để mô hình hóa agent xử lý ticket hỗ trợ khách hàng.

```
START -> intake -> classify --(route_after_classify)-->
  simple       -> answer -> finalize -> END
  tool         -> tool -> evaluate --(route_after_evaluate)-->
                            success     -> answer -> finalize -> END
                            needs_retry -> retry --(route_after_retry)-->
                                             attempt<max -> tool (loop)
                                             else        -> dead_letter -> finalize -> END
  missing_info -> clarify -> finalize -> END
  risky        -> risky_action -> approval --(route_after_approval)-->
                                    approved -> tool -> evaluate -> ...
                                    rejected -> clarify -> finalize -> END
  error        -> retry --(route_after_retry)--> tool / dead_letter
```

Tổng cộng có **11 node**: intake, classify, tool, evaluate, answer, clarify, risky_action,
approval, retry, dead_letter, finalize.
Và **4 router có điều kiện**: `route_after_classify`, `route_after_evaluate`,
`route_after_retry`, `route_after_approval`.

**Tích hợp LLM (cloud / Gemini):**
- `classify_node` — Dùng `ChatGoogleGenerativeAI.with_structured_output(Classification)`
  để phân loại intent một cách đáng tin cậy với thứ tự ưu tiên rõ ràng
  (risky > tool > missing_info > error > simple).
- `answer_node` — Gemini tạo câu trả lời cuối dựa trên `tool_results` + quyết định phê duyệt + câu hỏi gốc.
- `evaluate_node` — Đánh giá bằng LLM (tính năng bonus) cho kết quả tool không lỗi, kết hợp với
  kiểm tra chuỗi `"ERROR"` để đảm bảo vòng retry luôn hoạt động đúng.

## 3. Cấu trúc State

| Trường | Reducer | Lý do |
|---|---|---|
| thread_id, scenario_id, query | overwrite | định danh run / đầu vào |
| route, risk_level | overwrite | phân loại hiện tại |
| attempt, max_attempts | overwrite | bộ đếm retry có giới hạn |
| final_answer | overwrite | câu trả lời mới nhất |
| evaluation_result | overwrite | điều khiển `route_after_evaluate` |
| pending_question | overwrite | luồng làm rõ thông tin |
| proposed_action | overwrite | action rủi ro đang chờ phê duyệt |
| approval | overwrite | payload quyết định HITL |
| messages | append (`operator.add`) | lịch sử hội thoại/kiểm toán |
| tool_results | append | tích lũy output của tool qua các lần retry |
| errors | append | tích lũy lỗi tạm thời |
| events | append | log kiểm toán chỉ thêm (drives metrics) |

Các trường điều khiển vô hướng dùng overwrite để giữ checkpoint nhẹ; 4 channel dạng list
là append-only để lịch sử kiểm toán tồn tại qua các vòng retry.

## 4. Kết quả kiểm thử các kịch bản

| Chỉ số | Giá trị |
|---|---:|
| Tổng kịch bản | 24 |
| Tỷ lệ thành công | 100% |
| Trung bình node đã thăm | 9.04 |
| Tổng số retry | 33 |
| Tổng số interrupt (HITL) | 8 |
| Khả năng phục hồi sau crash | True |


| Kịch bản | Kỳ vọng | Thực tế | Kết quả | Retry | Interrupt | Yêu cầu/Quan sát phê duyệt | Độ trễ(ms) |
|---|---|---|:---:|---:|---:|:---:|---:|
| G01_simple | simple | simple | ✅ | 0 | 0 | False/False | 4469 |
| G02_simple_nokw | simple | simple | ✅ | 0 | 0 | False/False | 3139 |
| G03_simple_tricky | simple | simple | ✅ | 0 | 0 | False/False | 3603 |
| G04_tool | tool | tool | ✅ | 0 | 0 | False/False | 7244 |
| G05_tool_nokw | tool | tool | ✅ | 0 | 0 | False/False | 4968 |
| G06_tool_indirect | tool | tool | ✅ | 0 | 0 | False/False | 21011 |
| G07_missing | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3706 |
| G08_missing_subtle | missing_info | missing_info | ✅ | 0 | 0 | False/False | 4673 |
| G09_missing_oneword | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3692 |
| G10_risky_easy | risky | risky | ✅ | 3 | 1 | True/True | 6417 |
| G11_risky_indirect | risky | risky | ✅ | 3 | 1 | True/True | 6863 |
| G12_risky_polite | risky | risky | ✅ | 3 | 1 | True/True | 7084 |
| G13_risky_imperative | risky | risky | ✅ | 3 | 1 | True/True | 6963 |
| G14_risky_disguised | risky | risky | ✅ | 3 | 1 | True/True | 6186 |
| G15_error_easy | error | error | ✅ | 2 | 0 | False/False | 4059 |
| G16_error_nokw | error | error | ✅ | 2 | 0 | False/False | 2923 |
| G17_error_narrative | error | error | ✅ | 2 | 0 | False/False | 3380 |
| G18_dead | error | error | ✅ | 1 | 0 | False/False | 1239 |
| G19_priority_risky_vs_tool | risky | risky | ✅ | 3 | 1 | True/True | 8879 |
| G20_priority_risky_vs_simple | risky | risky | ✅ | 2 | 1 | True/True | 9016 |
| G21_priority_tool_vs_error | tool | tool | ✅ | 3 | 0 | False/False | 7659 |
| G22_priority_missing_vs_simple | missing_info | missing_info | ✅ | 0 | 0 | False/False | 3658 |
| G23_long_simple | simple | simple | ✅ | 0 | 0 | False/False | 4166 |
| G24_long_risky | risky | risky | ✅ | 3 | 1 | True/True | 7089 |


## 5. Phân tích các tình huống lỗi

1. **Tool lỗi tạm thời → retry có giới hạn.** `tool_node` mô phỏng lỗi tạm thời
   trên route `error`; `evaluate_node` phát hiện và điều hướng sang `retry`,
   tăng `attempt`. `route_after_retry` kiểm tra `attempt < max_attempts`
   nên vòng lặp có giới hạn — khi hết số lần thử sẽ chuyển sang `dead_letter`
   (xem kịch bản dead-letter với `max_attempts=1`).
2. **Action rủi ro không có phê duyệt.** Các yêu cầu refund/delete/email được phân loại
   `risky` và bắt buộc qua `risky_action -> approval` trước khi chạy bất kỳ tool nào.
   Nếu bị từ chối, luồng chuyển sang `clarify` thay vì thực thi side effect,
   đảm bảo action phá hủy không thể bỏ qua cổng human-in-the-loop.
3. **LLM API downtime (khả năng phục hồi).** Mỗi LLM node giảm dần về fallback
   deterministic khi có exception thay vì crash, nên lỗi API tạm thời không bao giờ
   làm abort grading trong khi LLM vẫn là đường dẫn có thẩm quyền.

## 6. Bằng chứng persistence / phục hồi

Mỗi run sử dụng `thread_id` riêng cho từng kịch bản (`thread-<scenario_id>`) và một
checkpointer. Backend `sqlite` (`persistence.py`) ghi state bền vững vào
`checkpoints*.db` (chế độ WAL), cho phép `get_state_history()` duyệt thời gian và
crash-resume. `recovery.verify_crash_resume()` chứng minh điều này: chạy một kịch bản,
hủy graph + saver, khởi tạo lại cả hai từ DB trên đĩa, và đọc state trở lại.
Chạy với `make demo-resume` (hoặc `agent-lab demo-resume`); kết quả được ghi
trong `reports/resume_evidence.md` và cờ metric `resume_success`
(hiện là `True`, 6 checkpoint đã phục hồi).

## 7. Các phần mở rộng đã thực hiện

- **SQLite persistence** (`persistence.py`): `SqliteSaver` bền vững với chế độ WAL.
- **HITL thực sự**: `LANGGRAPH_INTERRUPT=true` chuyển `approval_node` sang
  `langgraph.types.interrupt()` để pause/resume phê duyệt thực sự.
- **Đánh giá bằng LLM** trong `evaluate_node`.
- **Xuất Mermaid diagram** của graph đã biên dịch (`reports/graph.mermaid`,
  qua `make diagram`).
- **Time-travel / crash-resume** được demo qua SQLite checkpointer.
- **Dashboard chi phí token + reasoning** (`make visualize`): file HTML tự chứa
  `outputs/visualization.html` hiển thị từng bước, reasoning của từng bước,
  độ trễ, sử dụng token và chi phí USD ước tính cho mỗi kịch bản.

## 8. Kế hoạch cải thiện

Nếu có thêm một ngày: (1) thay `tool_node` mock bằng tool calls thực sau registry có kiểu
+ timeout riêng cho từng tool; (2) thêm `Send()` để gọi song song các lookup độc lập;
(3) thêm LangSmith tracing để quan sát độ trễ/chi phí; (4) chuyển approvals sang
durable queue để HITL tồn tại qua các lần restart; (5) thêm property-based tests
trên các query được tạo để củng cố bộ phân loại.
