# So sánh Agent-Radar và DAIN

## Bảng so sánh

| Tiêu chí | Agent-Radar | DAIN |
|---|---|---|
| **Mục tiêu** | Cải thiện cách tác tử LLM tận dụng lịch sử trao đổi dài | Cải thiện fusion đa phương thức cho suy luận cộng tác |
| **Loại bài toán** | Quản lý ngữ cảnh trong giao tiếp MAS dựa trên LLM | Fusion đa phương thức học được (phân loại/hồi quy) |
| **Kiến trúc** | Plug-in lớp chọn ngữ cảnh + lái attention đặt trên hệ MAS có sẵn (GPTSwarm, AutoGen, MAD) | Mạng nơ-ron hoàn chỉnh: Meta-Controller + pool 8 agent + đồ thị giao tiếp + consensus fusion |
| **Cơ chế giao tiếp** | Tác tử giữ nguyên cách giao tiếp; chỉ thay đổi *cách một tác tử đọc lại* lịch sử (attention steering) | Giao tiếp nén qua bottleneck có điều khiển: `m = φ·W_B h`, đồ thị Φ theo từng mẫu |
| **Cách chọn thông tin/tác tử** | Chọn **câu** theo score = cosine ngữ nghĩa × decay không gian (hop trên đồ thị) × decay thời gian (tuổi tin nhắn), ngưỡng θ | Chọn **agent** theo top-τ gating phụ thuộc ngữ cảnh; chọn **luồng thông tin** theo Φ = σ(G_comm(e_c)) |
| **Yêu cầu huấn luyện** | Training-free (chỉ cần sentence encoder pretrained) | Huấn luyện end-to-end với loss đa mục tiêu |
| **Chi phí** | +23% thời gian/task (968.8→1196.5s); không đổi trọng số LLM | 12.8M tổng params, 5.8M effective; tiết kiệm nhờ kích hoạt thưa |
| **Bộ dữ liệu** | HotpotQA, 2Wiki, MuSiQue, MATH-500, MMLU-Pro (văn bản) | ADNI, MIMIC-IV, MM-IMDB, CMU-MOSI, ENRICO (đa phương thức) |
| **Ưu điểm** | Không mất thông tin (giữ full transcript), không cần huấn luyện, cắm được vào nhiều MAS | Hiệu quả tính toán, khả năng diễn giải qua activation pattern, cân bằng đa mục tiêu |
| **Giới hạn** | Cần backend can thiệp attention (SPA/PASTA) → khó dùng với LLM qua API; cần chỉnh λ_s, λ_t, θ | Phân bổ agent thủ công; chỉ phân loại/hồi quy; cần huấn luyện từ đầu |

## Phân tích

### Điểm tương đồng thực sự

Cả hai paper cùng phản ứng với một vấn đề chung: khi số thành phần tham gia cộng tác tăng lên (nhiều tác tử, nhiều vòng, nhiều modality), việc để mọi thành phần trao đổi/truy cập *mọi thứ* vừa tốn kém vừa làm giảm chất lượng. Agent-Radar giải quyết ở mức *thông tin trong ngữ cảnh*: không phải mọi câu trong lịch sử đều đáng chú ý như nhau. DAIN giải quyết ở mức *thành phần tham gia tính toán*: không phải mọi agent đều cần chạy cho mỗi mẫu. Cả hai đều đi đến dạng **chọn lọc phụ thuộc ngữ cảnh** (context-dependent selection) thay vì chính sách tĩnh: Agent-Radar tính score cho từng câu theo truy vấn hiện tại, DAIN tính gate cho từng agent theo context vector của từng mẫu.

Điểm tương đồng thứ hai: cả hai đều dùng **cấu trúc đồ thị** để định lượng "khoảng cách". Agent-Radar đo khoảng cách hop giữa tác tử và suy giảm trọng số theo nó; DAIN học trực tiếp đồ thị giao tiếp Φ. Khác biệt là Agent-Radar coi topology là dữ kiện cho trước (cố định theo hệ MAS), còn DAIN coi topology là thứ cần học.

Điểm tương đồng thứ ba: cả hai đều muốn *không phá vỡ luồng hoạt động có sẵn*. Agent-Radar không sửa trọng số LLM hay topology của MAS; DAIN giữ nguyên backbone encoder và chỉ thêm phần điều phối.

### Khác biệt quan trọng

Thứ nhất, hai phương pháp nằm ở **tầng khác nhau của ngăn xếp**: Agent-Radar là kỹ thuật inference-time cho hệ đã có (LLM-based MAS), không có tham số cần huấn luyện; DAIN là một kiến trúc học từ đầu, toàn bộ hoạt động của nó được hình thành trong huấn luyện. Không thể nói phương pháp nào "tốt hơn" bằng cách so điểm: chúng không chạy trên cùng loại dữ liệu, cùng loại tác vụ, và không chia sẻ baseline nào.

Thứ hai, "động" (dynamic) mang hai nghĩa khác nhau. Trong Agent-Radar, động là việc ngữ cảnh được chọn lại ở *mỗi bước sinh của từng tác tử*. Trong DAIN, động là việc nhóm agent kích hoạt và đồ thị giao tiếp được tính lại cho *mỗi mẫu đầu vào*. Độ颗粒 của sự lựa chọn khác nhau: câu văn so với thành phần mô hình.

Thứ ba, chi phí: Agent-Radar *tăng* chi phí inference (~23%) để đổi lấy chất lượng, không giảm token đi vào prompt (toàn bộ lịch sử vẫn nằm trong prompt). DAIN *giảm* chi phí (effective params 5.8M/12.8M) vì thành phần không được chọn thì không chạy. Hai hướng tối ưu tài nguyên ngược chiều nhau.

### Bài học thiết kế chung

1. **Chọn lọc phụ thuộc ngữ cảnh thay vì chính sách tĩnh**: cả hai paper đều cho thấy lựa chọn nên được tính từ đặc điểm của từng tình huống (truy vấn, mẫu đầu vào) chứ không cố định trước.
2. **Khoảng cách cấu trúc là tín hiệu hữu ích**: khoảng cách hop (Agent-Radar) và đồ thị giao tiếp học được (DAIN) đều cải thiện chất lượng so với coi mọi nguồn như ngang hàng — được xác nhận bằng ablation ở cả hai paper (Table 4 của Agent-Radar: bỏ spatial decay làm MuSiQue giảm 39.72→28.41; Table 4 của DAIN: bỏ giao tiếp làm ADNI giảm 73.5→72.1).
3. **Nén thông tin có kiểm soát tốt hơn cắt cụt mù**: Agent-Radar giữ full transcript và chỉ lái attention; DAIN nén tin nhắn qua bottleneck tỷ lệ với độ mạnh kết nối thay vì chặn hẳn.
4. **Cần cơ chế tự cân bằng chi phí**: sparsity regularization của DAIN (‖g‖₁, ‖Φ‖²_F) và ngưỡng θ của Agent-Radar đều là "van điều tiết" giữa độ phủ và chi phí.

### Hướng kết hợp (chỉ là giả thuyết nghiên cứu, chưa được tác giả nào chứng minh)

Có thể hình dung một hệ trong đó DAIN-style meta-controller điều phối các agent LLM, và Agent-Radar-style decay quản lý ngữ cảnh mà mỗi agent LLM đọc lại giữa các vòng. Lưu ý đây chỉ là suy nghĩ thiết kế: hai paper chưa thử nghiệm kết hợp này, và hai cơ chế hoạt động ở tầng khác nhau (chọn thành phần vs. chọn ngữ cảnh), nên hiệu quả kết hợp là câu hỏi mở cần thí nghiệm riêng.
