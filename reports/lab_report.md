# Day 08 Lab Report — Điều Phối Agent LangGraph

## 1. Thông tin sinh viên

- Họ tên: Vũ Minh Duy
- Repo/commit: phase2-track3-day8-langgraph-agent (main)
- Nhà cung cấp LLM: Google Gemini (cloud) thông qua `langchain-google-genai`

## 2. Kiến trúc hệ thống

Một `StateGraph` với `AgentState` được định kiểu mô hình hóa một agent xử lý vé hỗ trợ.

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

**11 nút**: intake, classify, tool, evaluate, answer, clarify, risky_action,
approval, retry, dead_letter, finalize.
**4 bộ định tuyến có điều kiện**: `route_after_classify`, `route_after_evaluate`,
`route_after_retry`, `route_after_approval`.

**Tích hợp LLM (cloud / Gemini):**
- `classify_node` — `ChatGoogleGenerativeAI.with_structured_output(Classification)`
  để phân loại intent enum một cách đáng tin cậy với thứ tự ưu tiên rõ ràng
  (risky > tool > missing_info > error > simple).
- `answer_node` — Gemini tạo phản hồi cuối cùng dựa trên nghiêm ngặt
  `tool_results` + quyết định phê duyệt + truy vấn gốc.
- `evaluate_node` — Đánh giá bằng LLM (bonus) cho kết quả tool không lỗi, với
  cổng substring `"ERROR"` xác định để vòng lặp retry luôn đáng tin cậy.

## 3. Lược đồ trạng thái

| Trường | Reducer | Lý do |
|---|---|---|
| thread_id, scenario_id, query | overwrite | nhận dạng run / đầu vào |
| route, risk_level | overwrite | phân loại hiện tại |
| attempt, max_attempts | overwrite | bộ đếm retry giới hạn |
| final_answer | overwrite | câu trả lời mới nhất |
| evaluation_result | overwrite | điều khiển `route_after_evaluate` |
| pending_question | overwrite | luồng làm rõ |
| proposed_action | overwrite | hành động rủi ro chờ phê duyệt |
| approval | overwrite | payload quyết định HITL |
| messages | append (`operator.add`) | hội thoại/bản ghi kiểm tra |
| tool_results | append | tích lũy đầu ra tool qua các lần retry |
| errors | append | tích lũy lỗi tạm thời |
| events | append | log kiểm tra chỉ thêm (điều khiển metrics) |

Các trường điều khiển vô hướng dùng overwrite để giữ checkpoint lean; bốn
kênh list dùng append-only để bản ghi kiểm tra tồn tại qua các vòng retry.

## 4. Kết quả kịch bản

| Chỉ số | Giá trị |
|---|---:|
| Tổng kịch bản | 7 |
| Tỷ lệ thành công | 100% |
| TB nút đã thăm | 6.43 |
| Tổng retry | 3 |
| Tổng interrupt (HITL) | 2 |
| Khôi phục thành công | True |


| Kịch bản | Kỳ vọng | Thực tế | Thành công | Retry | Interrupt | Yêu cầu/Quan sát phê duyệt | Độ trễ(ms) |
|---|---|---|:---:|---:|---:|:---:|---:|
| S01_simple | simple | simple | ✅ | 0 | 0 | False/False | 3888 |
| S02_tool | tool | tool | ✅ | 0 | 0 | False/False | 3030 |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | False/False | 2596 |
| S04_risky | risky | risky | ✅ | 0 | 1 | True/True | 3589 |
| S05_error | error | error | ✅ | 2 | 0 | False/False | 1994 |
| S06_delete | risky | risky | ✅ | 0 | 1 | True/True | 2859 |
| S07_dead_letter | error | error | ✅ | 1 | 0 | False/False | 999 |


## 5. Phân tích lỗi

1. **Lỗi tool tạm thời → retry giới hạn.** `tool_node` mô phỏng lỗi tạm thời
   trên route `error`; `evaluate_node` phát hiện và định tuyến đến `retry`,
   tăng `attempt`. `route_after_retry` thực thi `attempt < max_attempts`
   để giới hạn vòng lặp — khi cạn kiệt sẽ chuyển đến `dead_letter`
   (xem kịch bản dead-letter với `max_attempts=1`).
2. **Hành động rủi ro không có phê duyệt.** Yêu cầu hoàn tiền/xóa/email được phân loại
   `risky` và bắt buộc qua `risky_action -> approval` trước khi tool chạy.
   Từ chối định tuyến đến `clarify` thay vì thực thi side effect, do đó
   hành động destructive không thể bỏ qua cổng human-in-the-loop.
3. **Sự cố LLM API (khả năng phục hồi).** Mỗi nút LLM giảm xuống fallback
   xác định khi có exception thay vì crash run, nên lỗi API tạm thời
   không bao giờ hủy grading trong khi LLM vẫn là đường dẫn có thẩm quyền.

## 6. Bằng chứng persistence / khôi phục

Mỗi run sử dụng `thread_id` theo kịch bản (`thread-<scenario_id>`) và
một checkpointer. Backend `sqlite` (`persistence.py`) ghi trạng thái bền vững vào
`checkpoints*.db` (chế độ WAL), cho phép `get_state_history()` duyệt thời gian và
crash-resume. `recovery.verify_crash_resume()` chứng minh điều này: chạy kịch bản,
bỏ graph + saver, tái tạo cả hai từ cùng DB trên đĩa, và đọc trạng thái lại.
Chạy với `make demo-resume` (hoặc `agent-lab demo-resume`); kết quả được ghi
trong `reports/resume_evidence.md` và cờ metric `resume_success`
(hiện tại `True`, 6 checkpoint đã khôi phục).

## 7. Công việc mở rộng

- **SQLite persistence** (`persistence.py`): `SqliteSaver` bền vững với chế độ WAL.
- **HITL thực sự**: `LANGGRAPH_INTERRUPT=true` chuyển `approval_node` sang
  `langgraph.types.interrupt()` cho phê duyệt pause/resume thực sự.
- **Đánh giá bằng LLM** trong `evaluate_node`.
- **Xuất biểu đồ Mermaid** của graph đã biên dịch (`reports/graph.mermaid`,
  qua `make diagram`).
- **Crash-resume / duyệt thời gian** được minh chứng qua SQLite checkpointer.
- **Bảng điều khiển chi phí token + reasoning** (`make visualize`): file HTML tự
  chứa `outputs/visualization.html` hiển thị mỗi bước, reasoning theo bước,
  độ trễ, sử dụng token và chi phí USD ước tính cho mỗi kịch bản.

## 8. Kế hoạch cải thiện

Với thêm một ngày: (1) thay `tool_node` mock bằng lời gọi tool thực qua
registry có kiểu + timeout cho từng tool; (2) thêm `Send()` fan-out song song
cho tra cứu độc lập; (3) thêm LangSmith tracing cho quan sát độ trễ/chi phí;
(4) chuyển approvals sang hàng đợi bền vững để HITL tồn tại qua restart; (5) thêm
test dựa trên thuộc tính cho truy vấn sinh ra để tăng cường phân loại.