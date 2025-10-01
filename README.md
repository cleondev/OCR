# OCR Service

Dịch vụ OCR mẫu được xây dựng cho hệ thống quản lý khoản vay. Mục tiêu là cung cấp API OCR mạnh mẽ có thể xử lý nhiều định dạng tài liệu (Word, PDF văn bản, PDF scan, ảnh) với hai động cơ lõi: Tesseract và PaddleOCR. Toàn bộ lịch sử các lần chạy OCR được lưu trữ trong SQLite cùng với các tệp trung gian để phục vụ việc kiểm tra, tái tạo.

## Tính năng chính

- **API FastAPI**: Endpoint REST để tải tài liệu và lựa chọn động cơ OCR (`tesseract` hoặc `paddle`).
- **Tiền xử lý ảnh**: Pipeline tự động gồm chuyển xám, cân bằng tương phản, lọc nhiễu, nhị phân hóa nhằm nâng cao độ chính xác OCR.
- **Hỗ trợ đa định dạng**: Ảnh (`png`, `jpg`, `jpeg`, `tiff`, `bmp`), PDF, Word (`doc`, `docx`). Word được chuyển sang PDF bằng LibreOffice để trích xuất ảnh trang.
- **Quản lý lịch sử**: Lưu vào SQLite thông tin tệp gốc, tệp chuyển đổi, ảnh gốc, ảnh tiền xử lý, văn bản OCR kèm độ tin cậy.
- **Lưu trữ tệp**: Toàn bộ tệp gốc, trang PDF, ảnh tiền xử lý được lưu trong thư mục `storage/` theo từng lần chạy.
- **Docker hóa**: Dockerfile cài đặt đầy đủ phụ thuộc (Tesseract, LibreOffice, Poppler, PaddleOCR) để triển khai nhất quán.
- **Tuỳ chọn ngôn ngữ theo từng lần chạy**: API và giao diện web cho phép nhập mã ngôn ngữ (`lang`) riêng cho từng động cơ, hỗ trợ PaddleOCR tiếng Việt (`vi`) cũng như các mã Tesseract (`vie+eng`, `eng`,...).

## Cấu trúc thư mục

```
app/
  config.py          # Cấu hình ứng dụng
  database.py        # Khởi tạo SQLite + SQLAlchemy session
  main.py            # FastAPI app và endpoints
  models.py          # Mô hình ORM
  schemas.py         # Schema Pydantic cho API
  services/
    file_processing.py  # Lưu file, chuyển đổi PDF/Word → ảnh
    preprocess.py       # Pipeline tiền xử lý ảnh
    ocr_base.py         # Định nghĩa giao diện OCR
    tesseract_engine.py # Triển khai động cơ Tesseract
    paddle_engine.py    # Triển khai động cơ PaddleOCR
    ocr_service.py      # Điều phối toàn bộ quy trình OCR
requirements.txt    # Các phụ thuộc Python
Dockerfile          # Docker hóa dịch vụ
```

## Thiết lập ngôn ngữ OCR

- Mặc định Tesseract chạy với cấu hình `vie+eng` để ưu tiên tiếng Việt nhưng vẫn giữ lại khả năng nhận diện tiếng Anh.
- PaddleOCR được cấu hình với mã ngôn ngữ `vi`.
- Bộ từ điển tiếng Việt mặc định của PaddleOCR dựa trên bảng chữ cái Latin nên thiếu nhiều ký tự có dấu (`ă`, `â`, `ơ`, `ư`,...).
  Dự án bổ sung tệp `app/resources/paddle_vi_dict.txt` để nạp vào PaddleOCR. Do mô hình Latin chỉ hỗ trợ tối đa 185 ký tự, tệp đã được tinh gọn về đúng kích thước này bằng cách thay thế các ký tự ít dùng bằng bảng chữ cái tiếng Việt mở rộng. Nếu vượt quá giới hạn, Paddle sẽ giải mã sai (xuất hiện ký tự `Ă` giữa các từ), vì vậy dịch vụ sẽ kiểm tra kích thước tệp và bỏ qua nếu không hợp lệ.
- Có thể thay đổi thông qua biến môi trường `OCR_TESS_LANG` và `OCR_PADDLE_LANG` trước khi khởi động dịch vụ.
- Ngoài cấu hình mặc định, mỗi lần gọi API `/api/v1/ocr` đều có thể truyền thêm tham số `lang` (ví dụ `lang=vi` khi sử dụng PaddleOCR). Giao diện web cũng có ô nhập ngôn ngữ và tự động gợi ý giá trị mặc định theo từng động cơ.
- Khi chạy bằng Dockerfile đi kèm, gói `tesseract-ocr-vie` đã được cài đặt sẵn để hỗ trợ tiếng Việt có dấu.

## Chạy dịch vụ cục bộ

### Yêu cầu hệ thống

- Docker và Docker Compose

### Các bước chạy

```bash
docker build -t ocr-service .
docker run -p 8000:8000 -v $(pwd)/storage:/app/storage ocr-service
```

Sau khi container chạy, API sẵn sàng tại `http://localhost:8000`. Truy cập `http://localhost:8000/docs` để thử nghiệm Swagger UI.

### Truy cập giao diện web

Sau khi container chạy, truy cập `http://localhost:8000/` để mở giao diện quản trị. Tại đây có thể tải tài liệu,
theo dõi lịch sử các phiên OCR và xem chi tiết từng trang ảnh/kết quả văn bản.

### Gọi thử API bằng `curl`

```bash
curl -X POST "http://localhost:8000/api/v1/ocr?engine=tesseract" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

## Lịch sử OCR

Các bảng chính trong SQLite (`storage/ocr.db`):

- `ocr_runs`: thông tin mỗi lần chạy (engine, đường dẫn tệp gốc, tệp chuyển đổi, văn bản tổng hợp tốt nhất, độ tin cậy).
- `ocr_images`: toàn bộ ảnh gốc và ảnh tiền xử lý, kèm nhãn và thứ tự.
- `ocr_text_results`: kết quả OCR cho từng biến thể ảnh, kèm confidence.

Có thể kiểm tra lịch sử qua API:

- `GET /api/v1/ocr`: Danh sách tất cả các lần chạy (mới nhất trước).
- `GET /api/v1/ocr/{id}`: Chi tiết một lần chạy.

## Ghi chú triển khai

- Pipeline tiền xử lý được thiết kế để cải thiện khả năng đọc của Tesseract/PaddleOCR, có thể mở rộng thêm các bước khác (ví dụ deskew, loại bỏ nhiễu nâng cao).
- Nếu xử lý khối lượng lớn, cân nhắc thêm hàng đợi (Celery/RQ) và lưu trữ ngoài (S3, MinIO) cho các tệp trung gian.
- PaddleOCR yêu cầu nhiều tài nguyên hơn so với Tesseract; trong Dockerfile đã cấu hình để sử dụng CPU.

## Giấy phép

Dự án nghiên cứu mẫu – tùy chỉnh theo nhu cầu thực tế.
