# Hướng Dẫn Sinh Mã LaTeX cho GitHub Copilot

Sử dụng bộ quy tắc này làm hướng dẫn cơ bản cho GitHub Copilot để đảm bảo mã $\LaTeX$ được sinh ra đạt chất lượng, tuân thủ quy tắc biên dịch một tệp duy nhất, và đáp ứng yêu cầu về bảng dài (multi-page tables).

## 1. Quy Tắc Môi Trường LaTeX Cơ Bản

Copilot phải tuân thủ các quy tắc sau khi tạo cấu trúc tài liệu $\LaTeX$:

### 1.1 Lớp Tài Liệu (Document Class)
- Luôn sử dụng `article` với kích thước font 12pt và khổ giấy a4paper
- Đối với báo cáo dài: sử dụng `report` class

```latex
\documentclass[12pt, a4paper]{article}
% hoặc
\documentclass[12pt, a4paper]{report}
```

### 1.2 Ngôn Ngữ và Font
- **Bắt buộc** sử dụng `babel` và `fontenc` để hỗ trợ tiếng Việt
- Sử dụng font Times New Roman hoặc Computer Modern (mặc định)
- Luôn thiết lập ngôn ngữ `vietnamese` là chính

```latex
\usepackage[T5]{fontenc}
\usepackage[vietnamese]{babel}
\usepackage[utf8]{inputenc}
```

### 1.3 Thiết Lập Trang và Kích Thước
- Cài đặt lề trang rõ ràng để đảm bảo nội dung căn chỉnh chuẩn xác
- Sử dụng `geometry` để thiết lập lề trên/dưới $2.5\text{cm}$, lề trái/phải $2\text{cm}$

```latex
\usepackage[left=2cm,right=2cm,top=2.5cm,bottom=2.5cm]{geometry}
```

### 1.4 Các Gói Bắt Buộc
Luôn thêm các gói sau vào preamble:

```latex
\usepackage{amsmath}       % Toán học
\usepackage{booktabs}      % Đường kẻ bảng đẹp hơn
\usepackage{longtable}     % Hỗ trợ bảng dài
\usepackage{tabularx}      % Bảng tự động điều chỉnh
\usepackage{array}         % Mở rộng tính năng bảng
\usepackage{multirow}      % Gộp dòng trong bảng
\usepackage{graphicx}      % Hình ảnh
\usepackage{float}         % Vị trí hình ảnh/bảng
\usepackage{hyperref}      % Liên kết
\usepackage{url}           % URL
\usepackage{enumitem}      % Tùy chỉnh danh sách
\usepackage{fancyhdr}      % Header/Footer
\usepackage{lipsum}        % Text mẫu (chỉ để test)
```## 2. Quy Tắc Sinh Bảng Dài (Multi-Page Tables)

Đây là quy tắc **quan trọng nhất**. Tất cả các bảng có thể kéo dài nội dung phải sử dụng môi trường `longtable` thay vì `tabular` hoặc `table` thông thường.

### 2.1 Template Bảng Bắt Buộc

Copilot phải mô phỏng chính xác cấu trúc sau cho mọi bảng phân tích hoặc bảng liệt kê chi tiết:

#### Quy tắc kích thước:
- Sử dụng tổng chiều rộng cột là $14.5\text{cm}$ (ví dụ: `p{3.5cm}` và `p{11cm}`) 
- Đảm bảo bảng căn chỉnh với lề tài liệu ($17\text{cm}$ tổng chiều rộng)
- Phần tiêu đề (header) của bảng phải được lặp lại ở đầu mỗi trang tiếp theo
- Phần chân bảng (footer) phải hiển thị thông báo "Tiếp theo ở trang sau" hoặc tương tự#### Template Bảng Chi Tiết:

```latex
\begin{longtable}{|p{3.5cm}|p{11cm}|}
    \caption{Tiêu đề Bảng (Ví dụ: Bảng phân tích Use Case: Tên Use Case)}
    \label{tab:ten-bang-de-truy-cap} \\
    \hline
    \textbf{Mục} & \textbf{Mô tả} \\ \hline
    \endfirsthead
    
    \hline
    \hline
    \textbf{Mục} & \textbf{Mô tả} \\ \hline
    \endhead
    
    \hline
    \endfoot
    
    \hline
    \endlastfoot
    % --- Nội dung Bảng bắt đầu từ đây ---
    \textbf{Tên Mục} & Nội dung chi tiết của mục đó. \\ \hline
    \textbf{Mục khác} & Mô tả chi tiết cho mục khác. \newline 
    Lưu ý: Sử dụng \texttt{\textbackslash newline} để xuống dòng trong một ô. \\ \hline
    \textbf{Mục có nội dung dài} & 
    1. Điểm thứ nhất \newline
    2. Điểm thứ hai \newline
    3. Điểm thứ ba với nội dung rất dài có thể tự động wrap xuống dòng \\ \hline
    % --- Kết thúc Nội dung Bảng ---
\end{longtable}
```

### 2.2 Các Loại Bảng Khác

#### Bảng ngắn (1 trang):
```latex
\begin{table}[H]
\centering
\begin{tabular}{|p{3.5cm}|p{11cm}|}
\hline
\textbf{Mục} & \textbf{Mô tả} \\ \hline
Mục 1 & Nội dung ngắn \\ \hline
Mục 2 & Nội dung ngắn khác \\ \hline
\end{tabular}
\caption{Tiêu đề bảng ngắn}
\label{tab:bang-ngan}
\end{table}
```

#### Bảng so sánh (3+ cột):
```latex
\begin{longtable}{|p{4cm}|p{5cm}|p{5cm}|}
    \caption{Bảng so sánh} \\
    \hline
    \textbf{Tiêu chí} & \textbf{Phương án A} & \textbf{Phương án B} \\ \hline
    \endfirsthead
    
    \hline
    \multicolumn{3}{|c|}{\textit{(Tiếp theo từ trang trước)}} \\
    \hline
    \textbf{Tiêu chí} & \textbf{Phương án A} & \textbf{Phương án B} \\ \hline
    \endhead
    
    \hline
    \multicolumn{3}{|r|}{\textit{(Tiếp tục ở trang sau...)}} \\
    \endfoot
    
    \hline
    \endlastfoot
    
    Tiêu chí 1 & Ưu điểm A & Ưu điểm B \\ \hline
    Tiêu chí 2 & Nhược điểm A & Nhược điểm B \\ \hline
\end{longtable}
```

### 2.3 Căn Chỉnh Chiều Ngang

- Đảm bảo tổng chiều rộng của các cột trong `longtable` được tính toán để căn chỉnh sát với lề text đã định nghĩa trong `geometry` (khoảng 17cm)
- **KHÔNG** sử dụng môi trường `\centering` cho `longtable` 
- Căn chỉnh bằng cách xác định chiều rộng cột (`p{...}`)
- Tổng chiều rộng cột = Chiều rộng trang - Lề trái - Lề phải = 21cm - 2cm - 2cm = 17cm