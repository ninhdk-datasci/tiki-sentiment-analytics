import joblib
import streamlit as st
import numpy as np
import time
import re
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Tiki Sentiment Analysis", layout="wide", page_icon="🛍️")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&family=Playfair+Display:wght@700;800&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }

html, body, .stApp {
    height: 100vh;
    overflow: hidden;
    background: #F8F6F3;
    font-family: 'Inter', sans-serif;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #E8E5E0; border-radius: 10px; }
::-webkit-scrollbar-thumb { background: #C0BCB5; border-radius: 10px; }

[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"], .block-container {
    padding: 0 !important;
    height: 100vh !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important; 
    background: linear-gradient(135deg, #0F0E0C 0%, #1A1814 100%);
}

[data-testid="stSidebarContent"] { background: transparent !important; padding: 0 !important; }

[data-testid="stHorizontalBlock"] { height: 100vh; gap: 0 !important; margin: 0 !important; }

[data-testid="column"]:nth-child(1) {
    background: #F8F6F3;
    border-right: 1px solid #E8E5E0;
    padding: 1.8rem 2rem !important;
    height: 100vh;
    overflow-y: auto;
}

[data-testid="column"]:nth-child(2) {
    background: #FFFFFF;
    padding: 1.8rem 2rem !important;
    height: 100vh;
    overflow-y: auto;
}

/* Textarea */
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1.5px solid #E8E5E0 !important;
    border-radius: 16px !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.9rem !important;
    color: #1A1814 !important;
    padding: 1rem 1.2rem !important;
    line-height: 1.6 !important;
    min-height: 180px !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.1) !important;
    outline: none !important;
}

[data-testid="stTextArea"] label { display: none !important; }

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
    color: white;
    border: none;
    border-radius: 40px;
    padding: 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 0.75rem 0 1.5rem 0;
    transition: all 0.25s ease;
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(255,107,53,0.35);
}

/* Result pill */
.result-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 24px;
    border-radius: 40px;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
}
.pill-pos { background: rgba(46, 139, 86, 0.12); color: #2E8B57; border: 1px solid rgba(46, 139, 86, 0.3); }
.pill-neu { background: rgba(212, 168, 39, 0.12); color: #B8860B; border: 1px solid rgba(212, 168, 39, 0.3); }
.pill-neg { background: rgba(220, 53, 69, 0.12); color: #DC3545; border: 1px solid rgba(220, 53, 69, 0.3); }

/* Panel headers */
.panel-header {
    display: inline-block;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #FF6B35;
}
.panel-header span {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF6B35;
}

/* Distribution box */
.dist-box {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.2rem;
    margin-top: 1rem;
    border: 1px solid #E8E5E0;
}
.dist-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF6B35;
    margin-bottom: 1rem;
}
.dist-item { margin-bottom: 0.8rem; }
.dist-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #6B6660;
    margin-bottom: 5px;
}
.dist-bar {
    height: 6px;
    background: #E8E5E0;
    border-radius: 10px;
    overflow: hidden;
}
.dist-fill { height: 100%; border-radius: 10px; }
.dist-stats {
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid #E8E5E0;
}
.dist-stat { text-align: center; }
.dist-stat-value { font-size: 1rem; font-weight: 700; color: #1A1814; }
.dist-stat-label { font-size: 0.6rem; color: #A5A099; }

/* Processed text box */
.processed-box {
    background: #F8F6F3;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    border-left: 3px solid #FF6B35;
}
.processed-box strong { color: #FF6B35; font-size: 0.7rem; text-transform: uppercase; }
.processed-box p {
    font-size: 0.75rem;
    color: #6B6660;
    line-height: 1.5;
    margin-top: 6px;
}

/* Placeholder */
.result-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 60%;
    gap: 15px;
    color: #C5C0B8;
}
.result-placeholder .emoji { font-size: 3rem; opacity: 0.5; }
.result-placeholder p { font-size: 0.75rem; letter-spacing: 1.5px; text-transform: uppercase; }

/* Alert */
.stAlert { border-radius: 16px !important; font-size: 0.8rem !important; border: none !important; }

/* Sidebar */
.sidebar-content { padding: 1.5rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
.sidebar-header .logo { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 800; color: #FF8C42; }
.sidebar-header .sub { font-size: 0.55rem; color: #8A867F; margin-top: 4px; }
.divider { height: 1px; background: #2D2A26; margin: 1rem 0; }
.section-title { font-size: 0.55rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #FF6B35; margin-bottom: 0.75rem; }
.stat-card { margin-bottom: 0.75rem; }
.stat-value { font-size: 1rem; font-weight: 700; color: #FFFFFF; }
.stat-label { font-size: 0.5rem; color: #8A867F; margin-top: 2px; }
.sidebar-footer { font-size: 0.5rem; color: #4A4742; text-align: center; padding-top: 1rem; border-top: 1px solid #2D2A26; }

/* Progress bar custom */
.stProgress > div > div { background-color: #FF6B35 !important; border-radius: 20px !important; }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_model():
    return joblib.load("sentiment_pipeline.pkl")

pipeline = load_model()

# ==================== PREPROCESSOR ====================
class TextPreprocessor:
    def __init__(self):
        self.slang_dict = {
            "k": "không", "ko": "không", "kh": "không", "hok": "không",
            "dc": "được", "đc": "được", "sp": "sản phẩm", "shop": "cửa hàng",
            "mk": "mình", "mik": "mình", "t": "tôi", "toi": "tôi",
            "ok": "tốt", "oki": "tốt", "oke": "tốt", "dep": "đẹp", "xau": "xấu",
            "nhanhg": "nhanh", "cham": "chậm", "tks": "cảm ơn", "vs": "với",
            "j": "gì", "r": "rồi", "nma": "nhưng mà", "nhma": "nhưng mà",
            "tot": "tốt", "khong": "không", "duoc": "được"
        }
        self.stopwords = [
            "a lô","ai","ai đó","alô","anh","ba","bao giờ","bao nhiêu","bây giờ",
            "bên","bạn","bản thân","bất cứ","bất kỳ","bằng","bởi","bởi vì","cho",
            "cho nên","cho đến","chung","chúng ta","chúng tôi","chỉ","chị","các",
            "cách","cái","câu hỏi","còn","có","có thể","cùng","cũng","của","cứ",
            "do","do đó","dù","dù sao","dùng","em","giờ","gì","gần","gồm","hay",
            "hoặc","hãy","hơn","họ","khi","khoảng","khác","kể","là","làm","lên","sao",
            "lúc","lại","lấy","mà","mình","mọi","mỗi","một","mới","nay","ngay",
            "nghe","nghĩ","ngoài","ngày","người","nhau","nhà","như","nhưng","những",
            "nào","này","nên","nếu","nói","nơi","nữa","qua","ra","rằng","rồi",
            "sau","so với","sẽ","số","sự","tại","theo","thì","thế","thôi",
            "thường","thực ra","tin","trong","trên","trước","tuy","tuy nhiên","từ",
            "tự","và","vài","vào","vì","vì vậy","với","vừa","về","vậy","xem",
            "xin","yêu cầu","đang","đâu","đây","đã","đó","được","đến","đều",
            "để","đối với","ở","shop","tiki","hàng","sản phẩm","đơn hàng",
            "đóng gói","giao hàng","nhé","nha","ạ","ơi","vâng","hì","hi","chứ","tiện","việc","nhỉ"
        ]
        model_name = "bmd1905/vietnamese-correction-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.correction_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def normalize_slang(self, text):
        return " ".join([self.slang_dict.get(w, w) for w in text.split()])

    def predict_correction(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = self.correction_model.generate(**inputs, max_length=128)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def remove_stopwords(self, review):
        return " ".join([w for w in review.strip().split() if w not in self.stopwords])

    def preprocess(self, review):
        review = review.lower()
        review = self.normalize_slang(review)
        review = self.predict_correction(review)
        review = review.strip().replace("\n", " ")
        review = re.sub(r"[^\w\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", review)
        review = re.sub(r"[^\w\s]", " ", review)
        review = re.sub(r"\s+", " ", review)
        review = re.sub(r"(.)\1{2,}", r"\1", review)
        review = ViTokenizer.tokenize(review)
        review = self.remove_stopwords(review)
        return review.lower()

@st.cache_resource
def load_preprocessor():
    return TextPreprocessor()

preprocessor = load_preprocessor()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <div>
            <div class="sidebar-header">
                <div class="logo">🛍️ TikiSent</div>
                <div class="sub">Vietnamese Sentiment Analysis</div>
            </div>
            <div class="divider"></div>
            <div class="section-title">📊 Mô hình</div>
            <div class="stat-card"><div class="stat-value">LinearSVC</div><div class="stat-label">Thuật toán</div></div>
            <div class="stat-card"><div class="stat-value">TF-IDF</div><div class="stat-label">Vector hóa</div></div>
            <div class="divider"></div>
            <div class="section-title">🎯 Hiệu suất</div>
            <div class="stat-card"><div class="stat-value">85.2%</div><div class="stat-label">F1-Score (macro)</div></div>
            <div class="stat-card"><div class="stat-value">84K+</div><div class="stat-label">Reviews huấn luyện</div></div>
        </div>
        <div class="sidebar-footer">⚡ 3-class classification<br>Pos · Neu · Neg</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================
label_map = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
icon_map = {0: "😞", 1: "😐", 2: "😊"}
full_icon = {0: "💔 Tiêu cực", 1: "🤔 Trung lập", 2: "🎉 Tích cực"}
bar_colors = ["#DC3545", "#D4A827", "#2E8B57"]

dist_data = [
    ("😊 Tích cực", 42000, 50, "#2E8B57"),
    ("😐 Trung lập", 18000, 21.4, "#D4A827"),
    ("😞 Tiêu cực", 24000, 28.6, "#DC3545")
]

col_left, col_right = st.columns(2, gap="large")

# ===== LEFT COLUMN =====
with col_left:
    st.markdown('<div class="panel-header"><span>✍️ NHẬP BÌNH LUẬN</span></div>', unsafe_allow_html=True)
    
    text = st.text_area(
        "review_input",
        placeholder="Ví dụ: Sản phẩm rất tốt, giao hàng nhanh, đóng gói cẩn thận...",
        label_visibility="collapsed",
        key="review_input",
        height=180
    )
    
    predict_btn = st.button("🔍 PHÂN TÍCH NGAY", use_container_width=True)
    
    # PHÂN BỔ DỮ LIỆU
    st.markdown('<div class="dist-box"><div class="dist-title">📊 PHÂN BỔ DỮ LIỆU HUẤN LUYỆN</div>', unsafe_allow_html=True)
    
    for name, count, pct, color in dist_data:
        st.markdown(f"""
        <div class="dist-item">
            <div class="dist-header"><span>{name}</span><span>{count:,} ({pct}%)</span></div>
            <div class="dist-bar"><div class="dist-fill" style="width:{pct}%; background:{color};"></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="dist-stats">
            <div class="dist-stat"><div class="dist-stat-value">84,000</div><div class="dist-stat-label">Tổng số</div></div>
            <div class="dist-stat"><div class="dist-stat-value">3 lớp</div><div class="dist-stat-label">Phân loại</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== RIGHT COLUMN =====
with col_right:
    st.markdown('<div class="panel-header"><span>📊 KẾT QUẢ PHÂN TÍCH</span></div>', unsafe_allow_html=True)
    
    if predict_btn and text.strip():
        with st.spinner("🔄 Đang xử lý bình luận..."):
            time.sleep(0.2)
            processed = preprocessor.preprocess(text)
            scores = pipeline.decision_function([processed])[0]
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
            pred = pipeline.predict([processed])[0]
        
        # Result pill
        pill_class = ["pill-neg", "pill-neu", "pill-pos"][pred]
        st.markdown(f'<div class="result-pill {pill_class}">{icon_map[pred]} <strong>{full_icon[pred]}</strong></div>', unsafe_allow_html=True)
        
        # Xác suất dự đoán - DÙNG ST.PROGRESS ĐỂ TRÁNH LỖI HTML
        st.markdown("#### 📈 Xác suất dự đoán")
        
        for i, (lbl, p, color) in enumerate(zip(label_map.values(), probs, bar_colors)):
            col1, col2 = st.columns([1, 4])
            with col1:
                if i == pred:
                    st.markdown(f"✅ **{icon_map[i]} {lbl}**")
                else:
                    st.markdown(f"   {icon_map[i]} {lbl}")
            with col2:
                st.progress(float(p), text=f"{p*100:.1f}%")
        
        st.markdown("---")
        
        # Biểu đồ tròn
        fig = go.Figure(data=[go.Pie(
            labels=list(label_map.values()),
            values=probs,
            marker=dict(colors=bar_colors),
            hole=0.4,
            textinfo='label+percent',
            textposition='auto',
            showlegend=False,
            pull=[0.05 if i==pred else 0 for i in range(3)]
        )])
        fig.update_layout(
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig, use_container_width=True, key="prob_chart")
        
        # Văn bản sau xử lý
        st.markdown(f'''
        <div class="processed-box">
            <strong>🔧 Văn bản sau xử lý</strong>
            <p>{processed[:280]}{"..." if len(processed) > 280 else ""}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.caption(f"✅ Đã phân tích lúc: {time.strftime('%H:%M:%S - %d/%m/%Y')}")
        
    elif predict_btn:
        st.warning("⚠️ Vui lòng nhập bình luận trước khi phân tích!")
    else:
        st.markdown("""
        <div class="result-placeholder">
            <div class="emoji">🔮</div>
            <p>Kết quả phân tích sẽ hiển thị tại đây</p>
            <p style="font-size:0.65rem;">Nhập bình luận và bấm "Phân tích ngay"</p>
        </div>
        """, unsafe_allow_html=True)