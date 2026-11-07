# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026
- **Runtime Colab:** CPU (Intel Xeon)
- **Python:** 3.13.18
- **PyTorch:** 2.11.0+cpu
- **Ultralytics:** 8.4.145
- **Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`
- **Ngưỡng detection/segmentation:** 0.35
- **Thay đổi so với notebook nguồn:** Không có thay đổi

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. `class_id`=`468`; `class_name`=`cab`; `rank`=`1`; `score`=`0.510915`; `taxonomy_name`=`ImageNet-1K`.
2. Record mô tả dự đoán cho **toàn bộ bức ảnh**, không khoanh một vật thể riêng. Phân loại ảnh trả về các lớp tổng thể (top 5); ảnh traffic có nhiều phương tiện nhưng chỉ một lớp được xếp hạng cao nhất.
3. Danh sách lớp đến từ taxonomy đi kèm checkpoint, ở đây là `ImageNet-1K`; model không tự tạo tên lớp mới cho từng ảnh.
4. `class_id` là mã lớp ổn định, `class_name` giúp người đọc hiểu lớp đó, còn `taxonomy_name` cho biết hệ lớp mà mã/tên này thuộc về. Ghi đủ giúp diễn giải và đối chiếu đúng taxonomy.
5. Guideline cần quy định chủ thể hoặc chủ đề ưu tiên, cách xử lý ảnh có nhiều vật/chủ đề và tiêu chí chọn đúng một lớp.
6. `score` chỉ biểu thị mức độ model nghiêng về dự đoán đó; nó không chứng minh nhãn đúng theo quy tắc của con người. Ground truth phải được gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Một record ở ngưỡng `0.35`: `class_name`=`person`; `score`=`0.912625`; `bbox_xyxy`=`[385.33, 69.24, 498.92, 348.92]`; `bbox_width`=`113.58`; `bbox_height`=`279.68`.
2. Box có góc trên-trái tại khoảng `(385.33, 69.24)` và góc dưới-phải tại khoảng `(498.92, 348.92)`, theo pixel. Box rộng khoảng `113.58` pixel và cao khoảng `279.68` pixel.
3. Theo kết quả so sánh trong notebook: ngưỡng `0.20` có `17` prediction, `0.35` có `11`, và `0.60` có `6`.
4. Khi hạ ngưỡng, reviewer phải rà soát thêm nhiều box hơn để xác định box sai, trùng, sai lớp hoặc bỏ sót; không nên xem score cao là bảo đảm đúng.
5. Quy tắc box: vẽ hình chữ nhật sát nhất có thể quanh phần nhìn thấy của đối tượng, không bao gồm nền không thuộc đối tượng.
6. Ở mép trái ảnh có một người chỉ xuất hiện một phần (tay/cánh tay bị cắt bởi khung ảnh). Guideline nên nêu rõ cách gán đối tượng bị cắt mép, chẳng hạn chỉ vẽ phần nhìn thấy; nếu không rõ có tính phần cơ thể đó là đối tượng riêng hay không thì escalation để thống nhất thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. `instance_id`=`kitchen-001`; `class_name`=`person`; `score`=`0.899318`; `polygon_point_count`=`348`; 5 điểm đầu của `polygon_xy`=`[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
2. Box chỉ cho biết hình chữ nhật bao quanh; polygon mô tả đường biên hình dạng của đối tượng, nhờ đó thể hiện phần nào là vật thể và phần nào là nền.
3. `instance_id` phân biệt từng đối tượng cụ thể trong một ảnh để liên kết mask/polygon với đúng instance. Nó không phải class ID (mã loại đối tượng) hay tracking ID (định danh theo dõi đối tượng qua nhiều khung hình).
4. Quy tắc mask: vẽ biên theo phần nhìn thấy của đúng instance, tránh lấn sang nền hoặc đối tượng bên cạnh; phần bị che không được tự suy diễn.
5. Các chiếc bát ở khu vực bàn phía dưới bên trái nằm gần/chồng lấp nhau nên đường biên giữa chúng khó xác định. Annotator nên tách mask theo phần biên nhìn thấy; nếu không thể xác định ranh giới nhất quán theo guideline thì escalation để Lab Coach quyết định.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho toàn ảnh, kèm taxonomy/lớp theo quy ước dự án | Ảnh traffic có nhiều phương tiện nhưng top-1 là `cab` (score `0.510915`); prediction không xác nhận lớp đúng | Chọn một lớp theo guideline và chủ đề ưu tiên; không dùng score làm ground truth | Đúng lớp/taxonomy chưa, và quy tắc chọn một lớp có được áp dụng nhất quán không |
| Phát hiện vật thể | Mỗi đối tượng: class và bounding box | Hạ ngưỡng từ `0.60` xuống `0.20` làm số prediction tăng từ `6` lên `17`; người ở mép ảnh bị cắt | Gán các đối tượng nhìn thấy theo quy tắc, vẽ box và escalation ca mơ hồ | Đúng/sai lớp, box có ôm sát đối tượng, có box thừa/trùng hoặc bỏ sót không |
| Instance segmentation | Mỗi instance: class và polygon/mask riêng | Các bát gần nhau khiến biên khó tách; polygon người có `348` điểm nhưng vẫn cần kiểm tra đường biên | Tạo mask cho từng instance theo phần nhìn thấy, không tự đoán phần bị che | Biên mask, phần lấn nền/instance khác, tính đầy đủ và nhất quán giữa các instance |

## 5. An toàn dữ liệu

1. Chỉ sử dụng dữ liệu được phép cho bài lab, không đưa dữ liệu cá nhân hoặc dữ liệu không được phép lên nơi lưu trữ/công cụ công cộng.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, dừng xử lý và báo Lab Coach/giảng viên để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

### Checklist nộp bài GitHub

[x] `REPORT.md` (file này đã điền)

[x] `day1_lab_outputs/classification_predictions.json`

[x] `day1_lab_outputs/detection_predictions.json`

[x] `day1_lab_outputs/segmentation_predictions.json`

[x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`

[x] `day1_lab_outputs/visuals/classification_top5.png`

[x] `day1_lab_outputs/visuals/detection_predictions.png`

[x] `day1_lab_outputs/visuals/segmentation_prediction.png`

[x] Ô validation cuối notebook báo `PASS`

[x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output

[x] Đã nộp **link repository** trên LMS
