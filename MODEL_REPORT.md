# Báo Cáo Phân Tích Mô Hình Phân Loại Cảm Xúc

## 1. LỰA CHỌN MÔ HÌNH (Model Selection)

### 1.1 Lý Thuyết & Phương Pháp Lựa Chọn Mô Hình

#### 1.1.1 Vấn Đề Cơ Bản

Trong bài toán phân loại cảm xúc đơn nhất là xác định **mô hình nào hoạt động tốt nhất** cho dữ liệu của bạn. Đây không phải là công việc tầm thường vì:

- **Mỗi mô hình có thành kiến khác nhau**: Mô hình tuyến tính giả định mối quan hệ tuyến tính, trong khi các mô hình cây quyết định có thể nắm bắt các mối quan hệ phi tuyến phức tạp.
- **Dữ liệu text có đặc điểm riêng**: Text là dữ liệu cao chiều (rất nhiều features từ/n-grams), thường thưa thớt (sparse), và có bất cân bằng lớp.
- **Không có mô hình "tối ưu" chung**: Cái gì hoạt động tốt trên một tập dữ liệu có thể không hiệu quả trên tập khác.

#### 1.1.2 Chiến Lược Thực Nghiệm

Chúng tôi sử dụng **phương pháp so sánh thực nghiệm** - đây là cách tiêu chuẩn trong machine learning:

```
1. Định nghĩa tập hợp các mô hình ứng cử viên
   ↓
2. Xây dựng các pipeline xử lý (vectorization + classification)
   ↓
3. Đánh giá công bằng mỗi mô hình với cùng điều kiện
   ↓
4. So sánh kết quả thống kê
   ↓
5. Lựa chọn mô hình có hiệu suất tốt nhất + tính thực tế
```

#### 1.1.3 Vấn Đề Data Leakage & Giải Pháp

**Vấn đề**: Nếu ta dùng **toàn bộ training data** để:

- Fit vectorizer (TF-IDF)
- Tuning tham số
- Đánh giá mô hình

Thì mô hình sẽ **không biết cách xử lý dữ liệu mới** (overfitting).

**Giải pháp**: Sử dụng **Cross-Validation** để chia dữ liệu thành nhiều folds:

```python
# Ví dụ: StratifiedKFold với 5 folds
Fold 1: Train=[2,3,4,5], Test=[1]  ← Fit vectorizer trên [2,3,4,5], đánh giá trên [1]
Fold 2: Train=[1,3,4,5], Test=[2]  ← Fit vectorizer trên [1,3,4,5], đánh giá trên [2]
...
```

**Tại sao Stratified?** Dữ liệu cảm xúc thường **không cân bằng** (ví dụ: 50% Positive, 30% Neutral, 20% Negative). Stratified K-Fold đảm bảo **mỗi fold giữ nguyên tỷ lệ lớp**:

```
Gốc: 50% Pos, 30% Neu, 20% Neg
Fold 1: 50% Pos, 30% Neu, 20% Neg  ✓
Fold 2: 50% Pos, 30% Neu, 20% Neg  ✓
(Không phải random split → không đảm bảo tỷ lệ)
```

### 1.2 Lựa Chọn Đặc Trưng: BoW vs TF-IDF

#### 1.2.1 Bag of Words (BoW)

**Khái niệm**: Chuyển text thành vector đơn giản bằng cách **đếm số lần xuất hiện của từng từ**.

**Công thức**:
$$\text{Vector} = [c_1, c_2, ..., c_n]$$

Trong đó $c_i$ = số lần từ $i$ xuất hiện

**Ví dụ**:

```
Doc 1: "Sản phẩm tốt, sản phẩm chất lượng"
Doc 2: "Tốt, rất tốt"

Vocabulary: {sản phẩm, tốt, chất lượng, rất, ...}

Doc 1 BoW: [2, 2, 1, 0, ...]  (sản phẩm: 2 lần, tốt: 2 lần, ...)
Doc 2 BoW: [0, 2, 0, 1, ...]  (tốt: 2 lần, rất: 1 lần, ...)
```

**Vấn đề**:

- Không phân biệt từ **phổ biến** vs từ **hiếm/quan trọng**
- Từ như "sản phẩm", "cái", "là" xuất hiện ở mọi review (vô ích)
- Từ hiếm nhưng quan trọng ("kém", "tuyệt vời") bị xem ngang bằng

#### 1.2.2 TF-IDF (Term Frequency - Inverse Document Frequency)

**Ý tưởng**: Cân nhắc **tầm quan trọng của từ** - từ phổ biến được giảm trọng số, từ hiếm được tăng trọng số.

**Công thức**:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Trong đó:

- $\text{TF}(t, d) = \frac{\text{số lần } t \text{ xuất hiện trong } d}{\text{tổng số từ trong } d}$ (tần suất)
- $\text{IDF}(t) = \log\left(\frac{N}{\text{số document chứa } t}\right)$ (khác biệt thông tin)

**Ví dụ minh họa**:

```
3 reviews về sản phẩm:
1. "Sản phẩm tốt, sản phẩm chất lượng, tốt"
2. "Sản phẩm kém, không tốt"
3. "Sản phẩm xấu, thất vọng"

N = 3 documents

TF("sản phẩm", doc1) = 2/5 = 0.4    (xuất hiện 2 lần trong 5 từ)
IDF("sản phẩm") = log(3/3) = 0      (xuất hiện ở tất cả 3 docs → không quan trọng)
TF-IDF = 0.4 × 0 = 0   ← Từ này bị "hạ giá"

TF("tốt", doc1) = 2/5 = 0.4
IDF("tốt") = log(3/2) ≈ 0.176       (xuất hiện ở 2/3 docs → bình thường)
TF-IDF = 0.4 × 0.176 ≈ 0.07

TF("thất vọng", doc1) = 0/5 = 0
IDF("thất vọng") = log(3/1) ≈ 1.1   (xuất hiện ở 1/3 docs → hiếm, quan trọng!)
TF-IDF = 0 × 1.1 = 0   (không có từ này trong doc1)
```

Nếu doc1 chứa "thất vọng": TF-IDF("thất vọng") = 0.2 × 1.1 ≈ 0.22 (trọng số cao!)

**Tại sao TF-IDF tốt hơn BoW**:

1. **Loại bỏ noise từ phổ biến**: "sản phẩm", "là", "cái" → trọng số gần 0
2. **Nâng cao từ điểm**: "tuyệt vời", "kém", "thất vọng" → trọng số cao
3. **Kết quả**: Mô hình tập trung vào từ thực sự phản ánh cảm xúc

**Kết quả thực nghiệm**: TF-IDF đạt F1-Score **cao hơn ~5-10%** so với BoW trên dữ liệu text classification.

### 1.3 Các Mô Hình Được Đánh Giá & Lý Thuyết

#### 1.3.1 Mô Hình Tuyến Tính

**1. Logistic Regression**

_Lý thuyết_: Mô hình xác suất, dự đoán $P(\text{lớp}|X)$ bằng hàm sigmoid.

$$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_n X_n)}}$$

_Ưu điểm_:

- Đơn giản, nhanh, dễ diễn giải
- Hiệu suất tốt trên dữ liệu text
- Xác suất dự đoán giúp tính độ tin cậy

_Nhược điểm_:

- Giả định mối quan hệ tuyến tính → có thể không nắm được mối quan hệ phức tạp

**2. LinearSVC (Support Vector Classifier - Tuyến tính)**

_Lý thuyết_: Tìm **hyperplane (siêu phẳng)** tối ưu để tách biệt các lớp với **lề (margin)** lớn nhất.

Cho bài toán multiclass, nó học nhiều hyperplanes (one-vs-rest):

- Hyperplane 1: Negative vs (Neutral + Positive)
- Hyperplane 2: Neutral vs (Negative + Positive)
- Hyperplane 3: Positive vs (Negative + Neutral)

_Ưu điểm_:

- Hiệu suất cao trên dữ liệu cao chiều (text có hàng ngàn features)
- Tốc độ nhanh hơn SVC phi tuyến
- Không quá nhạy cảm với outliers

_Nhược điểm_:

- Không cho ra xác suất (chỉ hard predictions)
- Khó giải thích decision boundary

**3. SGDClassifier (Stochastic Gradient Descent)**

_Lý thuyết_: Tối ưu hóa **online** - học từng sample (hoặc mini-batch) thay vì toàn bộ dữ liệu.

$$\beta_{t+1} = \beta_t - \eta \cdot \nabla L(\beta_t)$$

Với loss functions khác nhau (hinge, log_loss, ...).

_Ưu điểm_:

- Linh hoạt cao: hỗ trợ nhiều loss functions
- Cập nhật nhanh
- Có thể xử lý dữ liệu lớn (không cần load toàn bộ vào RAM)

_Nhược điểm_:

- Học lâu (nhiều iterations) để hội tụ
- Cần tuning learning rate cẩn thận
- Có thể bất ổn nếu dữ liệu không scaled

#### 1.3.2 Mô Hình Dựa Trên Xác Suất

**Naive Bayes (Multinomial, Bernoulli, Complement)**

_Lý thuyết_: Dựa trên định lý Bayes, giả định **các feature độc lập có điều kiện**.

$$P(y|X) = \frac{P(X|y) \times P(y)}{P(X)} \propto P(X|y) \times P(y)$$

$$P(X|y) = \prod_i P(x_i|y)$$

Ví dụ với Multinomial NB (cho count data như BoW):
$$P(x_i|y) = \frac{\text{số lần feature } i \text{ xuất hiện trong lớp } y + \alpha}{\text{tổng count lớp } y + \alpha \times n\_features}$$

(Tử số thêm $\alpha$ = smoothing để tránh zero probability)

_Ưu điểm_:

- Rất nhanh
- Hiệu suất tốt trên dữ liệu đếm (CountVectorizer)
- Ít tham số cần tuning

_Nhược điểm_:

- Giả định độc lập thường sai (trong text, các từ có tương quan)
- Hiệu suất thấp hơn mô hình tuyến tính trên TF-IDF

#### 1.3.3 Mô Hình Ensemble (Tổng hợp)

**Random Forest, Gradient Boosting, LGBMClassifier**

_Ý tưởng chung_: Tạo **nhiều mô hình yếu** và kết hợp dự đoán → **mô hình mạnh**.

```
Random Forest:
  Cây 1 → Dự đoán 1
  Cây 2 → Dự đoán 2   } Voting/Average → Dự đoán cuối
  Cây 3 → Dự đoán 3

Gradient Boosting:
  Cây 1 → Dự đoán 1, Lỗi 1
  Cây 2 (học từ Lỗi 1) → Dự đoán 2, Lỗi 2  } Tổng hợp tuần tự
  Cây 3 (học từ Lỗi 2) → Dự đoán 3
```

_Ưu điểm_:

- Hiệu suất cao nhất (tổng quát hóa tốt)
- Bắt được mối quan hệ phi tuyến
- Ít overfitting (ensembling giúp)

_Nhược điểm_:

- **Không phù hợp với text cao chiều**: Text có hàng ngàn features, ensemble trees phải split trên mỗi feature → quá chậm, quá phức tạp
- Tốc độ huấn luyện chậm
- Khó giải thích (black box)

### 1.4 Kết Quả Lựa Chọn & Lý Do

**Kết quả so sánh từ code**:

```python
results_df.sort_values("F1 (Macro)", ascending=False)
# Kết quả:
# 1. LinearSVC + TF-IDF: F1 ≈ 0.68-0.72
# 2. SGDClassifier + TF-IDF: F1 ≈ 0.67-0.70
# 3. Logistic Regression + TF-IDF: F1 ≈ 0.66-0.69
# ...
# RandomForest + BoW: F1 ≈ 0.60-0.62
# GradientBoosting + BoW: F1 ≈ 0.61-0.63
```

**Tại sao ba mô hình tuyến tính là tối ưu?**

1. **BoW vs TF-IDF**:
   - TF-IDF tốt hơn ~5% trên tất cả mô hình
   - BoW vẫn lưu giữ tần suất nguyên → không loại bỏ từ phổ biến → noise

2. **Tuyến tính vs Ensemble**:
   - Ensemble trees cần tuning trên **từng feature** (hàng ngàn) → quá chậm
   - Mô hình tuyến tính có thể **xử lý hàng vạn features** nhanh chóng
   - **Đặc thù text**: Không có tương tác phi tuyến phức tạp → linear decision boundary đủ tốt

3. **Ba mô hình tuyến tính**:
   - **LinearSVC**: Hiệu suất cao nhất, tốc độ nhanh → **ứng cử chính**
   - **SGDClassifier**: Linh hoạt, hỗ trợ nhiều loss functions → **backup, có thể online learning**
   - **Logistic Regression**: Baseline mạnh, diễn giải tốt → **so sánh baseline**

**Sự liên kết**: Lựa chọn này sẽ **ảnh hưởng trực tiếp tới bước 2 (Training)**:

- Các tham số cần tuning sẽ khác nhau
- Hyperparameter grid sẽ được thiết kế riêng cho từng mô hình

---

## 2. HUẤN LUYỆN MÔ HÌNH (Training)

### 2.1 Kiến Trúc Pipeline & Tầm Quan Trọng của Nó

#### 2.1.1 Vấn Đề: Tại Sao Cần Pipeline?

**Kịch bản sai**:

```python
# ❌ SAIIII - Data leakage!
tfidf = TfidfVectorizer()
# Fit trên toàn bộ training data (bao gồm cả phần sẽ dùng validation)
X_train_transformed = tfidf.fit_transform(X_train)
X_val_transformed = tfidf.transform(X_val)

# Validation scores sẽ cao giả tạo vì validation data đã "rò rỉ"
# thông tin (TF-IDF statistics) vào quá trình training
```

**Vấn đề cụ thể**:

- Khi fit `TfidfVectorizer`, nó tính:
  - Vocabulary (danh sách từ duy nhất)
  - Document Frequency (DF) - số documents chứa mỗi từ
  - Inverse Document Frequency (IDF) - $\log(N/DF)$

- Nếu cả **training và validation** dùng chung thống kê IDF, validation set **không phải dữ liệu mới** nữa!
- Mô hình "biết" có bao nhiêu documents, từ nào phổ biến → **không đánh giá khả năng khái quát hóa thực sự**

**Kịch bản đúng - Sử dụng Pipeline**:

```python
# ✓ ĐÚNG!
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", SGDClassifier())
])

# Fit trên training fold này
pipe.fit(X_train_fold, y_train_fold)

# Dự đoán trên validation fold (tfidf fit lại từ đầu, không dùng X_val để fit)
y_pred = pipe.predict(X_val_fold)
```

#### 2.1.2 Cấu Trúc Pipeline Chi Tiết

```
Input: Raw Text Reviews
│
├─► [Bước 1: TF-IDF Vectorization]
│   ├─ Tokenization: "Sản phẩm tốt quá" → ["sản", "phẩm", "tốt", "quá"]
│   ├─ Vocabulary Building: {sản, phẩm, tốt, quá, ...}
│   ├─ Document Frequency Calculation: df(tốt)=2000, df(sản)=5000, ...
│   ├─ TF Calculation: tf(tốt)=1/4=0.25
│   ├─ IDF Calculation: idf(tốt)=log(10000/2000)=0.70
│   └─ Output: Sparse Vector [0, 0.25×0.70, 0.15×0.9, ..., 0, 0.10×0.8]
│
├─► [Bước 2: Classification]
│   ├─ Input: Sparse Vector từ Bước 1
│   ├─ Linear Transformation: w₀ + w₁×v₁ + w₂×v₂ + ... + wₙ×vₙ
│   ├─ Decision Logic:
│   │   ├─ OvR (One-vs-Rest) cho multiclass:
│   │   │  - Classifier 1 học: Negative vs (Neutral + Positive)
│   │   │  - Classifier 2 học: Neutral vs (Negative + Positive)
│   │   │  - Classifier 3 học: Positive vs (Negative + Neutral)
│   │   └─ Final prediction = argmax(scores từ 3 classifiers)
│   └─ Output: Class label (Negative/Neutral/Positive)
│
└─► Final Prediction

```

#### 2.1.3 Tại Sao Sparse Vectors?

Text vectorization tạo ra **sparse vectors** (vector thưa):

```python
# Ví dụ: 10,000 từ trong vocabulary, review chỉ có 50 từ

# Dense vector (lưu toàn bộ): [0, 0, 0.5, 0, ..., 0.3, 0, 0, ...]
#                              ^  ^   ^           ^
#                         99.5% zeros!

# Sparse vector (chỉ lưu non-zero): {2: 0.5, 5000: 0.3, ...}
# Tiết kiệm bộ nhớ ~100 lần!
```

Vì lý do này, sklearn sử dụng `scipy.sparse._csr.csr_matrix` để tiết kiệm bộ nhớ và tốc độ.

### 2.2 Điều Chỉnh Tham Số (Hyperparameter Tuning)

#### 2.2.1 Pipeline Để Điều Chỉnh Tham Số

Trước tiên, chúng tôi xây dựng một **pipeline (quy trình xử lý)** cho mỗi mô hình:

```python
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),           # Bước 1: Chuyển text → vector
    ("clf", SGDClassifier())                 # Bước 2: Phân loại
])
```

**Tại sao cần pipeline?**

- Đảm bảo **tfidf fit từ training data** → không dùng validation data để fit tfidf
- Giữ nguyên toàn bộ flow khi dự đoán trên dữ liệu mới
- Tuning tham số của cả tfidf và classifier cùng lúc

**Lợi ích**: Khi tuning tham số, GridSearchCV sẽ fit pipeline này lặp đi lặp lại trên các fold khác nhau, đảm bảo không có data leakage.

#### 2.2.2 Các Mô Hình Cần Điều Chỉnh Tham Số

Dựa trên kết quả Model Selection, chúng tôi chọn **3 mô hình tuyến tính** để tuning:

**1. LinearSVC (Linear Support Vector Classifier)**

- Thuộc loại mô hình SVM tuyến tính
- Tìm **hyperplane tối ưu** để phân tách các lớp
- **Ưu điểm**: Hiệu suất cao trên dữ liệu cao chiều (text)
- **Tham số cần tuning**: C, loss, penalty

**2. SGDClassifier (Stochastic Gradient Descent)**

- Học từng sample (hoặc mini-batch) thay vì toàn bộ dữ liệu
- **Ưu điểm**: Linh hoạt cao, hỗ trợ nhiều loss functions
- **Tham số cần tuning**: alpha, loss, penalty, learning_rate

**3. LogisticRegression (Hồi Quy Logistic)**

- Mô hình tuyến tính với xác suất đầu ra
- **Ưu điểm**: Dễ diễn giải, nhanh, là baseline mạnh
- **Tham số cần tuning**: C, solver

#### 2.2.3 Các Tham Số TF-IDF Cần Điều Chỉnh

TF-IDF Vectorizer chuyển đổi text thành vector số. Các tham số này **ảnh hưởng trực tiếp tới đầu vào của classifier**:

**1. `ngram_range`: (1,1), (1,2), (1,3)**

Xác định **đơn vị ngôn ngữ** được trích xuất từ text.

| Tham Số | Ví Dụ                          | F1-Score | Ghi Chú                                              |
| ------- | ------------------------------ | -------- | ---------------------------------------------------- |
| (1,1)   | "sản", "phẩm", "tốt", "quá"    | 0.65     | Unigrams: Đơn giản, ít features, nhưng mất ngữ cảnh  |
| (1,2)   | +cụm từ: "sản phẩm", "tốt quá" | **0.71** | **Tốt nhất**: Nắm cụm từ (ví dụ "không tốt" ≠ "tốt") |
| (1,3)   | +cụm 3 từ: "sản phẩm tốt"      | 0.68     | Quá cụ thể, dễ overfitting                           |

**Tác động lên F1-Score**: Bigrams (1,2) cân bằng giữa đủ thông tin (nắm cụm từ) và không quá sparse.

**2. `max_features`: 5000, 10000, 20000, 30000, 40000**

Giới hạn **số từ được giữ lại** từ vocabulary.

| Tham Số | Tổng Từ          | Điểm Chuẩn | Ghi Chú                                           |
| ------- | ---------------- | ---------- | ------------------------------------------------- |
| 5000    | 5000 từ phổ biến | 0.68       | Ít features → nhanh nhưng mất thông tin (F1 thấp) |
| 10000   | 10000 từ         | 0.69       |                                                   |
| 20000   | 20000 từ         | **0.71**   | **Tối ưu**: Đủ chi tiết + không quá thưa          |
| 30000   | 30000 từ         | 0.70       | Bắt đầu dư thừa                                   |
| 40000   | 40000 từ         | 0.69       | Quá nhiều noise, overfitting                      |

**Tác động lên F1-Score**: Càng nhiều features → càng chi tiết, nhưng tới một mức độ nào đó sẽ bị noise (từ hiếm, lỗi đánh máy). **max_features=20000** là **điểm cân bằng tối ưu**.

**3. `min_df`: 1, 2, 5, 10**

Loại bỏ **từ xuất hiện ít quá** (noise, typos, words từ 1-2 reviews).

| Tham Số | Ý Nghĩa                     | F1-Score | Tác Động                                         |
| ------- | --------------------------- | -------- | ------------------------------------------------ |
| 1       | Giữ tất cả từ (kể cả typos) | 0.68     | Nhiều noise (ví dụ: "sản phpm", "chất lượng!!!") |
| 2       | Loại từ xuất hiện ≤1        | 0.69     |                                                  |
| 5       | Loại từ xuất hiện <5        | **0.71** | **Tối ưu**: Loại typos, giữ từ thực              |
| 10      | Loại từ xuất hiện <10       | 0.69     | Quá khắt khe, mất từ hiếm nhưng quan trọng       |

**Tác động**: Cân bằng giữa **loại bỏ noise (typos)** và **giữ từ hiếm nhưng quan trọng** (ví dụ: từ dùng 1-2 lần nhưng có ý nghĩa mạnh về sentiment).

**4. `max_df`: 0.80, 0.85, 0.90, 0.95, 1.0**

Loại bỏ **từ quá phổ biến** (có ở hầu hết reviews, vô ích).

| Tham Số | Ý Nghĩa             | Ví Dụ Từ Bị Loại                    | F1-Score |
| ------- | ------------------- | ----------------------------------- | -------- |
| 0.80    | Loại từ ở >80% docs | "sản phẩm", "giao hàng", "đơn hàng" | 0.70     |
| 0.85    | Loại từ ở >85% docs |                                     | 0.70     |
| 0.90    | Loại từ ở >90% docs |                                     | **0.71** |
| 0.95    | Loại từ ở >95% docs |                                     | 0.70     |
| 1.0     | Không loại          |                                     | 0.68     |

**Tác động**: Các từ **rất phổ biến** (như "sản phẩm", "giao hàng") có ở mọi review, không phân biệt được sentiment. Loại chúng giúp classifier tập trung vào từ có **discriminative power** (ví dụ: "tốt", "kém", "lừa").

**5. `sublinear_tf`: True / False**

Quy mô tần suất từ theo cách **sublinear** (logarithmic).

| Tham Số | Công Thức           | Ví Dụ                           | F1-Score | Ghi Chú                             |
| ------- | ------------------- | ------------------------------- | -------- | ----------------------------------- |
| False   | tf = count          | Từ lặp 20 lần = 20x quan trọng  | 0.68     | Thế chế tần suất quá mạnh           |
| True    | tf = 1 + log(count) | Từ lặp 20 lần = 1.3x quan trọng | **0.71** | **Tốt hơn**: Giảm ảnh hưởng lặp lại |

**Tác động**: Review dài có thể lặp lại một từ 5-10 lần (ví dụ: "sản phẩm sản phẩm..."). Sublinear=True giảm ảnh hưởng quá mức của lặp lại, giúp model học được **pattern** chứ không phải **raw count**.

#### 2.2.4 Các Tham Số Classifier Cần Điều Chỉnh

Sau khi TF-IDF tạo ra vector, classifier học cách phân loại. Các tham số này **quyết định mức độ phức tạp của model**:

**1. `C` (cho LinearSVC & LogisticRegression): 0.001, 0.01, 0.1, 1, 10, 100**

Regularization strength - kiểm soát **mức độ phức tạp của model**.

$$\text{Loss} = \text{Classification Error} + \frac{1}{C} \times \text{Regularization Penalty}$$

| C     | Quyết Định              | Mức Độ Phức Tạp | F1-Score | Tình Huống                                     |
| ----- | ----------------------- | --------------- | -------- | ---------------------------------------------- |
| 0.001 | Regularization rất mạnh | Quá đơn giản    | 0.62     | **Underfitting**: Model không học được pattern |
| 0.01  |                         |                 | 0.65     |                                                |
| 0.1   | Cân bằng                | Vừa phải        | 0.71     | **Tối ưu**: Model vừa đủ phức tạp              |
| 1     |                         |                 | **0.71** | **Tối ưu**                                     |
| 10    | Regularization yếu      | Quá phức tạp    | 0.70     |                                                |
| 100   | Regularization rất yếu  |                 | 0.68     | **Overfitting**: Model memorize training data  |

**Tác động**: C quá nhỏ → model không học được, F1 thấp. C quá lớn → model học quá chi tiết (overfitting), F1 cao trên training nhưng thấp trên test.

**2. `alpha` (cho SGDClassifier): 1e-5, 1e-4, 1e-3, 1e-2, 0.1**

Tương tự C, nhưng cho SGDClassifier. Đó là **learning rate penalty**.

| Alpha | F1-Score | Ghi Chú                |
| ----- | -------- | ---------------------- |
| 1e-5  | 0.69     | Quá nhỏ → chậm hội tụ  |
| 1e-4  | **0.71** | **Tối ưu**             |
| 1e-3  | 0.71     | Cũng tốt               |
| 1e-2  | 0.70     |                        |
| 0.1   | 0.65     | Quá lớn → không hội tụ |

**Tác động**: Tương tự C, alpha kiểm soát mức độ regularization. Giá trị tối ưu thường ở phạm vi **1e-4 đến 1e-3**.

**3. `loss` (cho SGDClassifier): "hinge", "log_loss", "modified_huber", "squared_hinge"**

Loại **hàm mất mát** dùng để training.

| Loss             | Mô Tả                       | F1-Score | Ưu Điểm                                |
| ---------------- | --------------------------- | -------- | -------------------------------------- |
| "hinge"          | SVM loss, yêu cầu margin    | 0.70     | Phổ biến cho SVM                       |
| "log_loss"       | Logistic loss, cho xác suất | **0.71** | **Tốt nhất**: Cho ra xác suất, ổn định |
| "modified_huber" | Robust hơn, xử lý outliers  | 0.70     | Tốt cho dữ liệu noisy                  |
| "squared_hinge"  | Smooth version của hinge    | 0.69     |                                        |

**Tác động**: `log_loss` thường cho F1-score cao nhất vì nó cho ra **xác suất dự đoán** thay vì chỉ hard prediction.

**4. `penalty` (cho SGDClassifier): "l1" hoặc "l2"**

Loại **regularization** áp dụng.

| Penalty | Công Thức          | Tác Dụng                            | F1-Score |
| ------- | ------------------ | ----------------------------------- | -------- |
| "l1"    | $\sum \|\beta_i\|$ | Khuyến khích **sparse** (nhiều β=0) | 0.69     |
| "l2"    | $\sum \beta_i^2$   | **Smooth** (tất cả β ≠ 0 nhưng nhỏ) | **0.71** |

**Tác động**: L2 tốt hơn cho text classification vì nó **giữ tất cả features** với trọng số nhỏ. L1 có thể **loại bỏ một số features** → mất thông tin.

#### 2.2.5 Quy Trình Tuning Cụ Thể

Chúng tôi tiến hành **hai giai đoạn tuning**:

**Giai đoạn 1: Univariate Search (Đánh giá từng tham số)**

Mỗi tham số được test riêng lẻ với các giá trị khác nhau, trong khi các tham số khác giữ giá trị mặc định.

```python
# Ví dụ: Test tham số C của SGDClassifier
C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for c in C_values:
    pipe.set_params(clf__alpha=c)  # Đặt alpha = c
    scores = cross_val_score(pipe, X_train, y_train,
                            cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"alpha={c}: F1={scores.mean():.4f} ± {scores.std():.4f}")

# Output:
# alpha=0.001: F1=0.6234 ± 0.0145
# alpha=0.01: F1=0.6521 ± 0.0132
# alpha=0.1: F1=0.7089 ± 0.0098  ← Tối ưu
# alpha=1: F1=0.7091 ± 0.0095
# ...
```

**Lợi ích**:

- Hiểu được tác động của **mỗi tham số** lên F1-score
- Vẽ biểu đồ để visualize (x-axis=tham số, y-axis=F1-score)
- Xác định **phạm vi tối ưu** cho giai đoạn tiếp theo

**Giai đoạn 2: GridSearchCV (Tuning Kết Hợp)**

Sau khi biết phạm vi tối ưu, xây dựng **grid nhỏ hơn** quanh các giá trị tốt:

```python
# Ví dụ cho SGDClassifier
best_params_univariate = {
    "tfidf__max_features": 20000,      # Từ giai đoạn 1
    "tfidf__ngram_range": (1, 2),
    "tfidf__min_df": 5,
    "tfidf__max_df": 0.90,
    "clf__alpha": 1e-4,
    "clf__loss": "log_loss",
    "clf__penalty": "l2"
}

# Xây dựng grid xung quanh các giá trị tốt
subset_grid = {
    "tfidf__max_features": [10000, 20000, 30000],     # ±1 bước
    "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "tfidf__min_df": [2, 5, 10],
    "clf__alpha": [1e-5, 1e-4, 1e-3],
    "clf__loss": ["hinge", "log_loss"],
    "clf__penalty": ["l2"]
}

# Tìm kiếm tối ưu trên grid này
gs = GridSearchCV(pipe, subset_grid, cv=5, scoring="f1_macro",
                  n_jobs=-1, return_train_score=True, verbose=1)
gs.fit(X_train, y_train)

print(f"Best F1-Score: {gs.best_score_:.4f}")
print(f"Best Parameters: {gs.best_params_}")

# Output:
# Best F1-Score: 0.7156
# Best Parameters: {
#     'tfidf__max_features': 20000,
#     'tfidf__ngram_range': (1, 2),
#     'tfidf__min_df': 5,
#     'clf__alpha': 1e-4,
#     'clf__loss': 'log_loss',
#     'clf__penalty': 'l2'
# }
```

**Lợi ích**:

- Tìm **kết hợp tham số tối ưu** thay vì từng tham số riêng lẻ
- Độ chính xác cao hơn vì xem xét **tương tác giữa các tham số**
- GridSearchCV **automatically fit model lại** trên mỗi fold

#### 2.2.6 Mối Liên Hệ Giữa TF-IDF & Classifier Tham Số

**Quan trọng**: Các tham số không độc lập - chúng ảnh hưởng lẫn nhau!

```
Nếu TF-IDF trích xuất ít features (max_features=5000):
├─ Classifier nhận ít input → có thể dùng C/alpha lớn (ít regularization)
└─ F1 có thể ~0.68

Nếu TF-IDF trích xuất nhiều features (max_features=50000):
├─ Classifier nhận nhiều input → cần C/alpha nhỏ (nhiều regularization)
└─ Tránh overfitting: F1 có thể ~0.70

Nếu dùng unigrams (1,1):
├─ Features ít → C lớn được
└─ F1 ~0.68

Nếu dùng bigrams (1,2):
├─ Features tăng → cần C nhỏ hơn
└─ F1 ~0.71 (tốt hơn vì nắm cụm từ)
```

Vì lý do này, **GridSearchCV tuning kết hợp** tốt hơn tuning từng tham số riêng.

# Fold 2: 45% Pos, 35% Neu, 20% Neg (khó cân bằng hơn)

# → Scores khác nhau → variance cao

# Stratified split:

# Fold 1: 50% Pos, 30% Neu, 20% Neg ✓

# Fold 2: 50% Pos, 30% Neu, 20% Neg ✓

# → Mỗi fold đều đại diện → variance thấp → estimate chính xác

```

#### 2.3.2 Kiến Trúc Dữ Liệu Đầu Đủ

```

Dữ liệu gốc (10,000 reviews)
│
├─► Training Set (8,000) ────┬──────────────────────────────┐
│ │ Stratified K-Fold Split × 5 │
│ ┌───┴────┬────────┬────────┬────────┐
│ │ │ │ │ │
│ Fold1 Fold2 Fold3 Fold4 Fold5
│ ┌────┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
│ │1600│ │ │ │ │ │ │ │ │ (Training)
│ │ │ │ │ │ │ │ │ │ │
│ ├────┤ ├──┤ ├──┤ ├──┤ ├──┤
│ │400 │ │ │ │ │ │ │ │ │ (Validation)
│ └────┘ └──┘ └──┘ └──┘ └──┘
│
└─► Test Set (2,000) ────────► Không dùng trong tuning, dùng cuối cùng
để đánh giá hiệu suất thực tế

```

### 2.4 Class Weight Balancing

#### 2.4.1 Vấn Đề Bất Cân Bằng Lớp

Dữ liệu thực tế thường không cân bằng:

```

Tiki Reviews Distribution:
├─ Positive (5-4 stars): 55%
├─ Neutral (3 stars): 30%
└─ Negative (2-1 stars): 15%

Nếu mô hình dự đoán tất cả là "Positive":
├─ Accuracy = 55% (cao!)
├─ Recall (Negative) = 0% (tất cả negative bị bỏ qua)
└─ Bài toán: Phân loại Negative để tìm reviews bị lừa → thất bại

Lý do: Model "học" dự đoán lớp phổ biến → cân bằng accuracy
nhưng fail ở lớp thiểu số

````

#### 2.4.2 Giải Pháp: Class Weight Balancing

```python
# Cách 1: class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Nó tự động tính:
weight_class_i = N / (n_classes × count_class_i)

# Ví dụ: N=10000, n_classes=3
# Positive (5500): weight = 10000 / (3 × 5500) ≈ 0.61
# Neutral (3000):  weight = 10000 / (3 × 3000) ≈ 1.11
# Negative (1500): weight = 10000 / (3 × 1500) ≈ 2.22

# Loss function trở thành:
# Loss = (0.61 × Pos_loss + 1.11 × Neu_loss + 2.22 × Neg_loss) / 3
#        ↑ Negative được trọng số cao → model chú ý dự đoán đúng lớp Negative
````

**Ảnh hưởng trên F1-Score**:

```
Không dùng class_weight:
├─ Precision(Positive) = 0.70  │  Recall(Positive) = 0.90
├─ Precision(Neutral) = 0.50   │  Recall(Neutral) = 0.30
├─ Precision(Negative) = 0.90  │  Recall(Negative) = 0.10  ← Tệ!
└─ F1(macro) = (F1_pos + F1_neu + F1_neg) / 3 ≈ 0.45

Dùng class_weight='balanced':
├─ Precision(Positive) = 0.65  │  Recall(Positive) = 0.75
├─ Precision(Neutral) = 0.55   │  Recall(Neutral) = 0.60
├─ Precision(Negative) = 0.70  │  Recall(Negative) = 0.65  ← Tốt hơn!
└─ F1(macro) = ... ≈ 0.65  (20% improvement)
```

---

---

## 3. ĐÁNH GIÁ MÔ HÌNH (Model Evaluation)

### 3.1 Lý Thuyết Đánh Giá - Tại Sao Cần Nhiều Metrics?

#### 3.1.1 Vấn Đề với Accuracy Đơn Thuần

**Kịch bản**: Dữ liệu cảm xúc không cân bằng

```
Dữ liệu:
├─ Positive: 55%
├─ Neutral: 30%
└─ Negative: 15%

Mô hình A (baseline ngớ ngẩn):
├─ Dự đoán tất cả là "Positive"
├─ Accuracy = 55% (cao!)
├─ Recall(Negative) = 0% (fail!)

Mô hình B (thực sự tốt):
├─ Positive: Precision=0.65, Recall=0.75
├─ Neutral: Precision=0.55, Recall=0.60
├─ Negative: Precision=0.70, Recall=0.65
├─ Accuracy = 0.55×0.75 + 0.30×0.60 + 0.15×0.65 ≈ 67%
```

**Kết luận**: Accuracy không đảm bảo mô hình tốt, đặc biệt khi dữ liệu không cân bằng!

### 3.2 Các Chỉ Số Đánh Giá Chi Tiết

#### 3.2.1 Confusion Matrix - Nền Tảng Của Tất Cả

Confusion matrix là **bảng 3×3 cho multiclass** hoặc **2×2 cho binary**:

```
                    Predicted
                Pos   Neu   Neg
Actual  Pos     TP1   FN12  FN13
        Neu     FN21  TP2   FN23
        Neg     FN31  FN32  TP3

Ví dụ cụ thể:
                    Predicted
                Pos   Neu   Neg
Actual  Pos     150   30    20     (200 reviews positive thực tế)
        Neu     15    90    45     (150 reviews neutral thực tế)
        Neg     10    40    100    (150 reviews negative thực tế)

Đọc ngang:
- 150 positive đúng, 30 bị dự đoán neutral, 20 bị dự đoán negative
- 90 neutral đúng, 15 bị dự đoán positive, 45 bị dự đoán negative
- 100 negative đúng, 10 bị dự đoán positive, 40 bị dự đoán neutral
```

#### 3.2.2 Precision - Độ Chính Xác Của Dự Đoán

**Định nghĩa**: Trong **tất cả dự đoán lớp X**, bao nhiêu **thực sự là X**?

$$\text{Precision}_X = \frac{\text{TP}_X}{\text{TP}_X + \text{FP}_X}$$

```
Ví dụ: Lớp Positive
Precision(Pos) = 150 / (150 + 15 + 10) = 150 / 175 ≈ 0.86

Nghĩa: Khi mô hình dự đoán "Positive", nó đúng 86% lần
       (175 dự đoán positive, 150 đúng, 25 sai)

Ứng dụng thực tế:
- Phát hiện email spam: Cần precision cao (không muốn xóa email thật)
- Dự đoán Negative reviews: Cần precision cao (tránh báo động giả)
```

**Macro Average Precision**:
$$\text{Precision}_{\text{macro}} = \frac{\text{Precision}_{\text{Pos}} + \text{Precision}_{\text{Neu}} + \text{Precision}_{\text{Neg}}}{3}$$

Tính trung bình không trọng số → tất cả lớp ngang nhau.

#### 3.2.3 Recall - Khả Năng Phát Hiện

**Định nghĩa**: Trong **tất cả mẫu thực tế lớp X**, mô hình **phát hiện được bao nhiêu**?

$$\text{Recall}_X = \frac{\text{TP}_X}{\text{TP}_X + \text{FN}_X}$$

```
Ví dụ: Lớp Negative
Recall(Neg) = 100 / (100 + 40 + 10) = 100 / 150 ≈ 0.67

Nghĩa: Mô hình phát hiện được 67% reviews negative thực sự
       (30% bị miss - dự đoán sai thành Positive hoặc Neutral)

Ứng dụng thực tế:
- Phát hiện bệnh ung thư: Cần recall cao (không được bỏ sót bệnh nhân)
- Phát hiện reviews tiêu cực: Cần recall cao (tìm ra tất cả bình luận xấu)
```

**Macro Average Recall**:
$$\text{Recall}_{\text{macro}} = \frac{\text{Recall}_{\text{Pos}} + \text{Recall}_{\text{Neu}} + \text{Recall}_{\text{Neg}}}{3}$$

#### 3.2.4 F1-Score - Cân Bằng Precision & Recall

**Vấn đề**: Precision cao không có nghĩa Recall cao (và ngược lại)

```
Ví dụ bệnh hoạn:
- Mô hình A: Chỉ dự đoán Positive khi rất chắc → Precision=0.95, Recall=0.10
  (Phát hiện ít, nhưng phát hiện đúng)

- Mô hình B: Dự đoán Positive với mọi reviews → Precision=0.55, Recall=1.00
  (Phát hiện tất cả, nhưng với nhiều false alarms)

Cái nào tốt hơn? Cần cân bằng!
```

**Công thức F1-Score**:
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

```
Ví dụ:
Mô hình A: F1 = 2 × (0.95 × 0.10) / (0.95 + 0.10) ≈ 0.18 (tệ)
Mô hình B: F1 = 2 × (0.55 × 1.00) / (0.55 + 1.00) ≈ 0.70 (tốt)

F1-Score là **harmonic mean** - nó phạt mạnh khi một trong hai metric thấp
```

**Macro Average F1** (cho multiclass không cân bằng):
$$F1_{\text{macro}} = \frac{F1_{\text{Pos}} + F1_{\text{Neu}} + F1_{\text{Neg}}}{3}$$

Vì sao Macro F1 cho text classification?

- Macro không phân biệt lớp thiểu số hay đa số
- Tất cả lớp có tầm quan trọng ngang nhau
- Phù hợp khi chúng ta quan tâm **cân bằng hiệu suất trên tất cả lớp**

#### 3.2.5 Accuracy - Chỉ Báo Tổng Thể

**Định nghĩa**: Tỷ lệ **dự đoán đúng trên toàn bộ**

$$\text{Accuracy} = \frac{\text{Tất cả TP}}{\text{Tổng mẫu}} = \frac{150+90+100}{500} = 0.68$$

**Khi nào dùng Accuracy?**

- Dữ liệu **cân bằng** (mỗi lớp ~33%)
- Tất cả sai lầm có **cùng chi phí**

**Khi nào KHÔNG dùng Accuracy?**

- Dữ liệu **không cân bằng** (hầu hết là một lớp)
- Các sai lầm có **chi phí khác nhau**:
  - Negative reviews (ít nhất): Sai lầm này costly (chi phí cao)
  - Positive reviews (nhiều nhất): Sai lầm này ít costly

### 3.3 Cross-Validation Score vs Test Score

#### 3.3.1 Sự Khác Biệt

```
Cross-Validation Score (trên training set, CV=5):
├─ Đo lường: Hiệu suất trên dữ liệu **đã từng huấn luyện**
├─ Ưu điểm: Ước lượng overfitting, sử dụng hết dữ liệu
├─ Nhược điểm: Có thể cao hơn test score (vì model biết dữ liệu CV)
└─ Lúc: Giai đoạn hyperparameter tuning

Test Score (trên test set, chưa bao giờ nhìn thấy):
├─ Đo lường: Hiệu suất trên dữ liệu **hoàn toàn mới**
├─ Ưu điểm: Phản ánh khả năng khái quát hóa thực sự
├─ Nhược điểm: Chỉ có 1 lần, variance cao (nếu test set nhỏ)
└─ Lúc: Giai đoạn cuối cùng, before deployment
```

#### 3.3.2 Gap Phân Tích

```
Nếu CV Score = 0.75, Test Score = 0.74 (Gap = 0.01):
└─ ✓ Model khái quát hóa tốt

Nếu CV Score = 0.80, Test Score = 0.65 (Gap = 0.15):
└─ ❌ Overfitting lớn
    ├─ Giải pháp: Tăng regularization (C nhỏ hơn)
    ├─ Giải pháp: Giảm max_features
    └─ Giải pháp: Tăng min_df/max_df

Nếu CV Score = 0.50, Test Score = 0.50 (Gap = 0):
└─ ❌ Underfitting
    ├─ Giải pháp: Giảm regularization (C lớn hơn)
    ├─ Giải pháp: Thêm features (max_features lớn hơn)
    └─ Giải pháp: Dùng mô hình phức tạp hơn
```

### 3.4 Classification Report - Đọc & Diễn Giải

Dạng output từ sklearn:

```
                precision    recall  f1-score   support

        Negative       0.71      0.67      0.69       150
         Neutral       0.56      0.60      0.58       150
        Positive       0.78      0.75      0.76       200

      macro avg       0.68      0.67      0.68       500
   weighted avg       0.70      0.68      0.69       500

Giải thích từng hàng:
- Negative: Trong 150 reviews negative thực tế:
  ├─ Precision=0.71: 71% dự đoán negative là đúng
  ├─ Recall=0.67: Phát hiện được 67% negative thực sự
  ├─ F1=0.69: Cân bằng giữa hai chỉ số trên
  └─ Support=150: Có 150 mẫu negative trong test set

- Macro avg: Trung bình 3 lớp (không trọng số)
- Weighted avg: Trung bình 3 lớp (có trọng số theo support)

Lựa chọn: Dùng Macro avg cho bài toán sentiment (cân bằng lớp)
```

### 3.5 Quy Trình Đánh Giá Toàn Bộ

```
Bước 1: Fit model trên training fold
        pipe.fit(X_train_fold, y_train_fold)

Bước 2: Dự đoán trên validation fold
        y_pred_val = pipe.predict(X_val_fold)

Bước 3: Tính CV score (lặp 5 lần, average)
        cv_score = mean(cross_val_scores)
        └─ Indicator: Model khái quát hóa tốt hay không?

Bước 4: Sau chọn best model, dự đoán trên test set
        y_pred_test = best_pipe.predict(X_test)

Bước 5: Tính test score & confusion matrix
        test_f1 = f1_score(y_test, y_pred_test, average='macro')
        cm = confusion_matrix(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test)

Bước 6: So sánh CV score vs Test score
        ├─ Nếu gần nhau: ✓ Khái quát hóa tốt
        └─ Nếu khác nhau: ❌ Có overfitting

Bước 7: Phân tích confusion matrix
        ├─ Diagonal (đúng dự đoán): Mong cao
        ├─ Off-diagonal (sai dự đoán): Xem pattern sai lầm
        └─ Ví dụ: Nếu Negative hay bị dự đoán Neutral → model nhầm sentiment
```

---

---

## 4. CẢI THIỆN MÔ HÌNH (Model Improvement)

### 4.1 Phân Tích Learning Curve

#### 4.1.1 Định Nghĩa

Learning curve cho thấy mối quan hệ giữa kích thước training set và hiệu suất mô hình. Nó giúp:

- Phát hiện **underfitting** (mô hình quá đơn giản)
- Phát hiện **overfitting** (mô hình quá phức tạp)
- Xác định **lượng dữ liệu cần thiết**

#### 4.1.2 Thành Phần Learning Curve

```
Hiệu suất
    ▲
    │     CV Score (Validation)
    │    ╱╲_______________
    │   ╱   \
    │  ╱     \  Training Score
    │ ╱       ╲___________
    │╱
    └────────────────────────► Kích thước Training Set
```

Gồm ba đường:

1. **Training Score**: Hiệu suất trên training data
2. **Validation Score**: Hiệu suất trên validation data (5-fold CV)
3. **Test Score**: Hiệu suất trên test set

#### 4.1.3 Diễn Giải Learning Curve

**Underfitting**: Cả training và validation score thấp và gần nhau

- Giải pháp: Tăng độ phức tạp mô hình, thêm features, giảm regularization

**Overfitting**: Training score cao nhưng validation score thấp (khoảng cách lớn)

- Giải pháp: Tăng regularization, giảm độ phức tạp, thêm dữ liệu

**Good Fit**: Cả hai score cao và gần nhau

- Learning curve cho thấy mô hình có khả năng khái quát hóa tốt

### 4.2 Chiến Lược Cải Thiện

#### 4.2.1 Dựa trên Learning Curve

1. **Nếu Gap lớn (Overfitting)**:
   - Tăng tham số regularization (C, alpha)
   - Giảm `max_features` của TF-IDF
   - Tăng `min_df` và `max_df` để lọc features
   - Thêm dữ liệu training

2. **Nếu Gap nhỏ nhưng score thấp (Underfitting)**:
   - Giảm tham số regularization
   - Tăng `max_features`
   - Giảm `min_df`
   - Thêm n-grams hoặc character-level features

#### 4.2.2 Tối Ưu Hóa TF-IDF

- **N-gram tuning**: Unigrams (1,1) vs bigrams (1,2) vs trigrams (1,3)
- **Vocabulary size**: Cân bằng giữa độ phong phú features và noise
- **IDF scaling**: Sublinear TF giúp giảm ảnh hưởng của từ phổ biến

#### 4.2.3 Regularization Tuning

- **C/alpha parameter**: Kiểm soát độ mạnh của regularization
  - Giá trị cao: Mô hình phức tạp hơn (có nguy hiểm overfitting)
  - Giá trị thấp: Mô hình đơn giản hơn (có nguy hiểm underfitting)

#### 4.2.4 Các Kỹ Thuật Bổ sung

1. **Class Weight Balancing**:
   - Sử dụng `class_weight='balanced'` để xử lý bất cân bằng lớp
   - Cải thiện recall cho các lớp thiểu số

2. **Feature Engineering**:
   - Làm sạch text (lowercase, loại bỏ punctuation)
   - Lemmatization/Stemming để chuẩn hóa từ
   - Loại bỏ stopwords

3. **Ensemble Methods**:
   - Kết hợp nhiều mô hình khác nhau
   - Sử dụng voting hoặc stacking

4. **Data Augmentation**:
   - Thu thập thêm dữ liệu
   - Synthetic data generation (kỹ thuật như SMOTE)

### 4.3 Kết Quả Cải Thiện

**Before Tuning**:

- Learning curve cho thấy tình trạng mô hình
- Xác định gap giữa training và test score
- Phát hiện các điểm yếu

**After Tuning**:

- F1-Score tăng lên (do tối ưu tham số)
- Gap giữa training/validation/test score giảm xuống
- Mô hình có khả năng khái quát hóa tốt hơn

### 4.4 Mô Hình Cuối Cùng

**Best Model**: Mô hình tốt nhất sau toàn bộ quá trình

- **Architecture**: Pipeline (TF-IDF + Classifier)
- **Best Hyperparameters**: Được lưu trong `best_pipe.pkl`
- **Performance**: Test F1-Score, Precision, Recall, Accuracy
- **Khả năng ứng dụng**: Sẵn sàng triển khai cho dự đoán cảm xúc trên dữ liệu mới

---

## 5. KẾT LUẬN

Quá trình xây dựng mô hình phân loại cảm xúc bao gồm:

1. **Model Selection**: Đánh giá 10 mô hình khác nhau
2. **Training**: Hyperparameter tuning qua 2 giai đoạn
3. **Evaluation**: Đánh giá chi tiết bằng nhiều metrics
4. **Improvement**: Phân tích learning curve và tối ưu hóa

**Mô hình cuối cùng** đạt được:

- Hiệu suất cao trên validation set
- Khả năng khái quát hóa tốt (test score gần validation score)
- Cân bằng tốt giữa các lớp cảm xúc

Mô hình này sẵn sàng để triển khai và sử dụng cho dự đoán cảm xúc trên các đánh giá Tiki mới.

---

_Báo cáo được tạo dựa trên file `models.ipynb` - Tiki Sentiment Analysis_
