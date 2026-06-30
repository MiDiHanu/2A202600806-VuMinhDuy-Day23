# Day 08 Lab Report

## 1. Thông tin sinh viên

- Họ tên: Vũ Minh Duy
- MSSV: 2A202600806
- Ngày nộp:

## 2. Thiết kế hệ thống

Mô tả các node, cạnh, trường state và reducer của bạn.

## 3. Cấu trúc State

Liệt kê các trường quan trọng và loại reducer tương ứng.

| Trường | Reducer | Lý do |
|---|---|---|
| messages | append | ghi nhận hội thoại/sự kiện |
| route | overwrite | chỉ route hiện tại |

## 4. Kết quả kịch bản

Dán các chỉ số chính từ `outputs/metrics.json`.

| Kịch bản | Route kỳ vọng | Route thực tế | Thành công | Retry | Interrupt |
|---|---|---|---:|---:|---:|

## 5. Phân tích lỗi

Mô tả ít nhất hai chế độ lỗi bạn đã xem xét:

1. Retry hoặc lỗi tool:
2. Action rủi ro không có phê duyệt:

## 6. Persistence / phục hồi

Giải thích cách bạn sử dụng checkpointer, thread id, state history, hoặc crash-resume.

## 7. Phần mở rộng

Mô tả bất kỳ phần mở rộng nào bạn đã hoàn thành: SQLite/Postgres, time travel, fan-out/fan-in, biểu đồ graph, tracing.

## 8. Kế hoạch cải thiện

Nếu bạn có thêm một ngày, điều gì bạn sẽ productionize trước?