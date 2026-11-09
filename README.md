# Dự Án Tốt Nghiệp - LúaAI: Hệ Thống Nhận Diện Bệnh & Sâu Hại Cây Lúa

Hệ thống ứng dụng học sâu (Deep Learning - CNN) và vi dịch vụ (Microservices) hỗ trợ bà con nông dân chẩn đoán sớm các bệnh và sâu hại phổ biến trên cây lúa, đồng thời cung cấp dự báo thời tiết nông vụ.

---

## 🌟 Tính Năng Chính
1. **Nhận diện bệnh lúa (Rice Disease Classification):**
   - Mô hình CNN fine-tuned trên nền tảng MobileNetV2.
   - Nhận diện 10 lớp trạng thái: *Healthy, bacterial_leaf_blight, brown_spot, leaf_blast, leaf_scald, narrow_brown_spot, neck_blast, rice_hispa, sheath_blight, tungro*.
2. **Nhận diện sâu hại lúa (Rice Pest Classification):**
   - Phân loại 6 nhóm sâu hại chính: *Brown Planthopper, Green Leafhoppers, Leaf Folders, Rice Bug, Stem Borer, Whorl-Maggot*.
3. **Dịch vụ thời tiết (Weather Service):**
   - Dự báo nhiệt độ, độ ẩm, lượng mưa khu vực canh tác (Sa Pa, Lào Cai) qua Open-Meteo API.
4. **Giao diện kiểm thử trực quan (Test Web Page):**
   - Giao diện web tải ảnh, xem trước và trả về kết quả chẩn đoán tức thì cùng tỷ lệ tin cậy (confidence score).

---

## 📂 Cấu Trúc Thư Mục

`	ext
├── models/                     # Cấu hình danh mục lớp và lịch sử huấn luyện
│   ├── class_names.json        # 10 lớp bệnh lúa
│   └── pest_class_names.json   # 6 lớp sâu hại
├── services/                   # Các vi dịch vụ (Microservices)
│   ├── ai_service/             # FastAPI phục vụ suy luận mô hình AI & Web UI
│   │   ├── main.py
│   │   └── test_page.html
│   └── weather_service/        # Dịch vụ thời tiết
│       └── main.py
├── training/                   # Pipeline xử lý dữ liệu và huấn luyện mô hình
│   ├── train_model.py          # Huấn luyện baseline MobileNetV2
│   ├── fine_tune_model.py      # Fine-tune mô hình bệnh lúa
│   ├── train_pest_mobilenetv2.py # Huấn luyện mô hình sâu hại
│   └── evaluate_finetuned.py   # Đánh giá độ chính xác, xuất ma trận nhầm lẫn
└── .gitignore                  # Bỏ qua virtualenv, dataset và file weights nặng (>100MB)
`

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+:
`ash
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

pip install tensorflow fastapi uvicorn pillow numpy httpx
`

### 2. Khởi chạy AI Service & Giao diện Test
`ash
python -m uvicorn services.ai_service.main:app --host 0.0.0.0 --port 8000 --reload
`
Truy cập trình duyệt tại:
- Giao diện chẩn đoán: http://localhost:8000/test
- Tài liệu API (Swagger UI): http://localhost:8000/docs

### 3. Khởi chạy Weather Service
`ash
python -m uvicorn services.weather_service.main:app --port 8001 --reload
`
