import streamlit as st
import torch
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")
import dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dotenv.load_dotenv()

st.set_page_config(
    page_title="VeriLang — Truth Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { margin:0; padding:0; box-sizing:border-box; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif !important; }

footer,header,#MainMenu { visibility:hidden !important; }
.stDeployButton { display:none !important; }
._container_gzau3_1,._profileContainer_gzau3_53,
._viewerBadge_nim44_23 { display:none !important; }
a[href*="streamlit.io"],
a[href*="share.streamlit.io"] { display:none !important; }
[data-testid="stToolbar"] { display:none !important; }

.stApp { background:#050914 !important; }

@keyframes gridMove {
    0%   { transform:translateY(0); }
    100% { transform:translateY(40px); }
}
.grid-bg {
    position:fixed; top:0; left:0;
    width:100%; height:100%;
    background-image:
        linear-gradient(rgba(99,179,237,0.04) 1px, transparent 1px),
        linear-gradient(90deg,rgba(99,179,237,0.04) 1px,transparent 1px);
    background-size:40px 40px;
    animation:gridMove 4s linear infinite;
    pointer-events:none; z-index:0;
}

@keyframes orb1 {
    0%,100% { transform:translate(0,0) scale(1); }
    50%     { transform:translate(40px,-30px) scale(1.1); }
}
@keyframes orb2 {
    0%,100% { transform:translate(0,0) scale(1); }
    50%     { transform:translate(-30px,40px) scale(0.9); }
}
.orb {
    position:fixed; border-radius:50%;
    filter:blur(80px); opacity:0.12;
    pointer-events:none; z-index:0;
}
.orb-1 {
    width:500px; height:500px; background:#3b82f6;
    top:-100px; left:-100px;
    animation:orb1 8s ease-in-out infinite;
}
.orb-2 {
    width:400px; height:400px; background:#8b5cf6;
    bottom:-100px; right:-100px;
    animation:orb2 10s ease-in-out infinite;
}
.orb-3 {
    width:300px; height:300px; background:#06b6d4;
    top:50%; left:50%;
    animation:orb1 12s ease-in-out infinite reverse;
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(30px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero {
    text-align:center; padding:50px 20px 36px;
    position:relative; z-index:1;
    animation:fadeUp 0.8s ease forwards;
}
.hero-tag {
    display:inline-block;
    background:rgba(59,130,246,0.15);
    border:1px solid rgba(59,130,246,0.4);
    color:#60a5fa; padding:6px 18px;
    border-radius:50px; font-size:13px;
    font-weight:600; letter-spacing:2px;
    text-transform:uppercase; margin-bottom:20px;
}
@keyframes titleGlow {
    0%,100% { text-shadow:0 0 40px rgba(59,130,246,0.3); }
    50%     { text-shadow:0 0 80px rgba(139,92,246,0.5),
                          0 0 120px rgba(59,130,246,0.3); }
}
.hero-title {
    font-size:72px; font-weight:700;
    background:linear-gradient(135deg,#ffffff 0%,#60a5fa 50%,#a78bfa 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1;
    animation:titleGlow 4s ease-in-out infinite;
    margin-bottom:14px;
}
.hero-sub {
    font-size:17px; color:rgba(255,255,255,0.5);
    max-width:540px; margin:0 auto 28px;
    line-height:1.7; font-weight:400;
}
.lang-pills {
    display:flex; justify-content:center;
    gap:10px; flex-wrap:wrap;
}
.lang-pill {
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    color:rgba(255,255,255,0.8);
    padding:5px 14px; border-radius:50px;
    font-size:13px; font-weight:500;
}

.stats-row {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:14px; margin:0 0 28px;
    position:relative; z-index:1;
}
.stat-box {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:18px;
    text-align:center; transition:all 0.3s;
    backdrop-filter:blur(10px);
}
.stat-box:hover {
    background:rgba(59,130,246,0.08);
    border-color:rgba(59,130,246,0.3);
    transform:translateY(-3px);
}
.stat-num {
    font-size:26px; font-weight:700;
    background:linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}
.stat-lbl {
    font-size:11px; color:rgba(255,255,255,0.35);
    margin-top:4px; font-weight:500;
    letter-spacing:0.5px;
}

.main-card {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:22px; padding:32px;
    position:relative; z-index:1;
    backdrop-filter:blur(20px);
    margin-bottom:20px;
}
.section-lbl {
    font-size:11px; font-weight:600;
    letter-spacing:2px; text-transform:uppercase;
    color:rgba(255,255,255,0.3); margin-bottom:10px;
}

.lang-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:10px; margin-bottom:22px;
}
.lang-tab {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:12px; padding:12px;
    text-align:center; transition:all 0.3s;
}
.lang-tab.active {
    background:rgba(59,130,246,0.15);
    border-color:rgba(59,130,246,0.5);
    box-shadow:0 0 20px rgba(59,130,246,0.15);
}
.lang-native {
    font-size:17px; font-weight:600; color:white;
}
.lang-eng {
    font-size:11px; color:rgba(255,255,255,0.35);
    margin-top:2px;
}

.stTextArea textarea {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:14px !important;
    color:rgba(255,255,255,0.9) !important;
    font-family:'Space Grotesk',sans-serif !important;
    font-size:15px !important; line-height:1.7 !important;
    transition:all 0.3s !important;
    caret-color:#60a5fa !important;
}
.stTextArea textarea:focus {
    border-color:rgba(59,130,246,0.5) !important;
    box-shadow:0 0 0 3px rgba(59,130,246,0.1) !important;
    background:rgba(59,130,246,0.04) !important;
}
.stTextArea textarea::placeholder {
    color:rgba(255,255,255,0.2) !important;
}

.stSelectbox > div > div {
    background:rgba(255,255,255,0.04) !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:12px !important; color:white !important;
}

@keyframes btnGlow {
    0%,100% { box-shadow:0 0 20px rgba(59,130,246,0.3); }
    50%     { box-shadow:0 0 40px rgba(139,92,246,0.5),
                         0 0 60px rgba(59,130,246,0.3); }
}
.stButton > button {
    background:linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:14px !important; padding:16px !important;
    font-size:16px !important; font-weight:600 !important;
    width:100% !important;
    font-family:'Space Grotesk',sans-serif !important;
    animation:btnGlow 3s ease-in-out infinite !important;
    transition:transform 0.2s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) scale(1.01) !important;
}

@keyframes resultIn {
    from { opacity:0; transform:translateY(20px) scale(0.98); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}
.result-fake {
    background:rgba(239,68,68,0.06);
    border:1px solid rgba(239,68,68,0.25);
    border-radius:20px; padding:28px; margin:16px 0;
    animation:resultIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;
    position:relative; overflow:hidden;
}
.result-fake::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#ef4444,#f97316);
}
.result-real {
    background:rgba(34,197,94,0.06);
    border:1px solid rgba(34,197,94,0.25);
    border-radius:20px; padding:28px; margin:16px 0;
    animation:resultIn 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;
    position:relative; overflow:hidden;
}
.result-real::before {
    content:''; position:absolute;
    top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#22c55e,#10b981);
}
.result-title-fake {
    font-size:26px; font-weight:700;
    color:#f87171; margin-bottom:6px;
}
.result-title-real {
    font-size:26px; font-weight:700;
    color:#4ade80; margin-bottom:6px;
}
.result-conf {
    font-size:13px; color:rgba(255,255,255,0.4);
    font-family:'JetBrains Mono',monospace;
}
.conf-track {
    height:8px; background:rgba(255,255,255,0.06);
    border-radius:4px; margin:14px 0 4px; overflow:hidden;
}
.conf-fill-fake {
    height:100%; border-radius:4px;
    background:linear-gradient(90deg,#ef4444,#f97316);
}
.conf-fill-real {
    height:100%; border-radius:4px;
    background:linear-gradient(90deg,#22c55e,#10b981);
}

.hinglish-badge {
    display:inline-block;
    background:rgba(251,191,36,0.1);
    border:1px solid rgba(251,191,36,0.3);
    color:#fbbf24; padding:4px 12px;
    border-radius:20px; font-size:12px;
    font-weight:600; margin-bottom:12px;
}

.meta-row {
    display:flex; gap:12px; margin-top:8px;
}
.meta-chip {
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:6px; padding:3px 10px;
    font-size:11px; color:rgba(255,255,255,0.35);
    font-family:'JetBrains Mono',monospace;
}

.fact-grid {
    display:grid; grid-template-columns:1fr 1fr;
    gap:10px; margin-top:14px;
}
.fact-link {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:10px 14px;
    font-size:12px; color:rgba(255,255,255,0.5);
    text-decoration:none; display:block;
    transition:all 0.2s;
}
.fact-link:hover {
    background:rgba(59,130,246,0.1);
    border-color:rgba(59,130,246,0.3);
    color:#60a5fa;
}

.warn-box {
    background:rgba(239,68,68,0.07);
    border:1px solid rgba(239,68,68,0.2);
    border-radius:12px; padding:14px 18px;
    margin-top:12px; font-size:13px;
    color:rgba(255,255,255,0.55); line-height:1.6;
}

.empty-state {
    background:rgba(255,255,255,0.02);
    border:1px dashed rgba(255,255,255,0.07);
    border-radius:20px; padding:60px 30px;
    text-align:center;
}

[data-testid="stSidebar"] {
    background:rgba(5,9,20,0.95) !important;
    border-right:1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * {
    color:rgba(255,255,255,0.7) !important;
}
.sb-logo {
    font-size:24px; font-weight:700;
    background:linear-gradient(135deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}
.sb-div {
    height:1px; background:rgba(255,255,255,0.06);
    margin:14px 0;
}
.acc-row {
    display:flex; justify-content:space-between;
    margin-bottom:4px;
}
.acc-lang { font-size:12px; color:rgba(255,255,255,0.5); }
.acc-val  { font-size:12px; font-weight:600; color:#60a5fa; }
.acc-track {
    height:4px; background:rgba(255,255,255,0.06);
    border-radius:2px; margin-bottom:10px;
}
.acc-fill {
    height:100%; border-radius:2px;
    background:linear-gradient(90deg,#3b82f6,#8b5cf6);
}
</style>

<div class="grid-bg"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI · NLP · Fact Detection</div>
    <div class="hero-title">VeriLang</div>
    <div class="hero-sub">
        Detect misinformation in WhatsApp forwards across
        Indian regional languages — now with Hinglish support
    </div>
    <div class="lang-pills">
        <span class="lang-pill">हिंदी</span>
        <span class="lang-pill">ગુજરાતી</span>
        <span class="lang-pill">मराठी</span>
        <span class="lang-pill">తెలుగు</span>
        <span class="lang-pill">Hinglish ✨</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-num">49K+</div>
        <div class="stat-lbl">Articles Trained</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">4+</div>
        <div class="stat-lbl">Indian Languages</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">99.92%</div>
        <div class="stat-lbl">Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">MuRIL</div>
        <div class="stat-lbl">Google AI Model</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    HF_REPO  = "maitry30/verilang-muril"
    hf_token = os.getenv("HF_TOKEN", None)
    tok = AutoTokenizer.from_pretrained(
        HF_REPO, token=hf_token
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        HF_REPO, token=hf_token,
        ignore_mismatched_sizes=True
    )
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
device = torch.device('cpu')
model  = model.to(device)

# ── Hinglish detection ────────────────────────────────────────
def is_romanized(text):
    ascii_c = sum(1 for c in text if ord(c) < 128)
    total   = len(text.replace(' ', ''))
    return (ascii_c/total > 0.75) if total > 0 else False

# ── Predict ───────────────────────────────────────────────────
def predict(text):
    inp = tokenizer(
        text, return_tensors='pt',
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        out   = model(**inp)
        probs = torch.softmax(out.logits, dim=1)
    fake_p = probs[0][0].item()
    real_p = probs[0][1].item()
    label  = "FAKE" if fake_p > real_p else "REAL"
    return label, max(fake_p,real_p), fake_p, real_p

# ── Layout ────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Select Language</div>',
                unsafe_allow_html=True)

    LANGS = {
        "Hindi":    ("हिंदी",    "Hindi"),
        "Gujarati": ("ગુજરાતી", "Gujarati"),
        "Marathi":  ("मराठी",   "Marathi"),
        "Telugu":   ("తెలుగు",  "Telugu")
    }

    language = st.selectbox(
        "lang", list(LANGS.keys()),
        label_visibility="collapsed"
    )

    cols = st.columns(4)
    for i, (lang, (native, eng)) in enumerate(LANGS.items()):
        active = "active" if lang==language else ""
        with cols[i]:
            st.markdown(f"""
            <div class="lang-tab {active}">
                <div class="lang-native">{native}</div>
                <div class="lang-eng">{eng}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-lbl">Paste WhatsApp Forward</div>',
                unsafe_allow_html=True)

    PLACEHOLDERS = {
        "Hindi":    "हिंदी में लिखें या Yeh khabar fake hai जैसे Hinglish में type करें...",
        "Gujarati": "ગુજરાતી માં લખો અથવા Aa khabar khoti chhe જેવું type કરો...",
        "Marathi":  "मराठी मध्ये लिहा किंवा Hi baatami khoti aahe असे type करा...",
        "Telugu":   "తెలుగులో రాయండి లేదా Ee news fake undi అని type చేయండి..."
    }

    user_input = st.text_area(
        "txt", placeholder=PLACEHOLDERS.get(language),
        height=190, label_visibility="collapsed"
    )

    if user_input:
        words  = len(user_input.split())
        chars  = len(user_input)
        script = "Hinglish/Roman" if is_romanized(user_input) \
                 else "Native Script"
        st.markdown(f"""
        <div class="meta-row">
            <span class="meta-chip">Words: {words}</span>
            <span class="meta-chip">Chars: {chars}</span>
            <span class="meta-chip">{script}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button(
        "⚡ Analyze for Misinformation", type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────
with right:
    if analyze_btn and user_input.strip():
        if len(user_input.split()) < 3:
            st.warning("Enter at least 3 words.")
        else:
            with st.spinner("Analyzing with MuRIL AI..."):
                time.sleep(0.3)
                label, conf, fake_p, real_p = predict(user_input)

            # Hinglish badge
            if is_romanized(user_input):
                st.markdown(
                    '<span class="hinglish-badge">'
                    '✨ Hinglish/Romanized Detected</span>',
                    unsafe_allow_html=True
                )

            # Result
            if label == "FAKE":
                st.markdown(f"""
                <div class="result-fake">
                    <div style="font-size:36px;margin-bottom:10px">⚠️</div>
                    <div class="result-title-fake">MISINFORMATION</div>
                    <div class="result-conf">
                        Confidence: {conf*100:.1f}% &nbsp;|&nbsp; {language}
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill-fake"
                             style="width:{conf*100:.0f}%"></div>
                    </div>
                    <div style="font-size:13px;
                                color:rgba(255,255,255,0.4);
                                margin-top:8px;line-height:1.6">
                        This text shows patterns found in
                        fake WhatsApp forwards.
                    </div>
                </div>
                <div class="warn-box">
                    Do NOT share this message. Verify from
                    trusted sources before forwarding.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-real">
                    <div style="font-size:36px;margin-bottom:10px">✅</div>
                    <div class="result-title-real">REAL NEWS</div>
                    <div class="result-conf">
                        Confidence: {conf*100:.1f}% &nbsp;|&nbsp; {language}
                    </div>
                    <div class="conf-track">
                        <div class="conf-fill-real"
                             style="width:{conf*100:.0f}%"></div>
                    </div>
                    <div style="font-size:13px;
                                color:rgba(255,255,255,0.4);
                                margin-top:8px;line-height:1.6">
                        This text shows patterns consistent
                        with credible news.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Chart
            st.markdown(
                '<div class="section-lbl" style="margin-top:18px">'
                'Probability Breakdown</div>',
                unsafe_allow_html=True
            )
            fig, ax = plt.subplots(figsize=(5, 2))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            ax.barh(
                ['Real', 'Fake'],
                [real_p*100, fake_p*100],
                color=['#22c55e','#ef4444'],
                height=0.45, edgecolor='none'
            )
            for i, prob in enumerate([real_p, fake_p]):
                ax.text(
                    prob*100+1, i,
                    f'{prob*100:.1f}%',
                    va='center', fontsize=10,
                    color='white', fontweight='600'
                )
            ax.set_xlim(0, 115)
            ax.tick_params(colors='white', labelsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(left=False, bottom=False)
            plt.tight_layout()
            st.pyplot(fig, transparent=True)

            # Fact check
            st.markdown("""
            <div class="section-lbl" style="margin-top:16px">
                Verify from trusted sources
            </div>
            <div class="fact-grid">
                <a class="fact-link"
                   href="https://pib.gov.in/factcheck.aspx"
                   target="_blank">🏛 PIB Fact Check</a>
                <a class="fact-link"
                   href="https://www.boomlive.in"
                   target="_blank">💥 Boom Live</a>
                <a class="fact-link"
                   href="https://www.altnews.in"
                   target="_blank">🔍 Alt News</a>
                <a class="fact-link"
                   href="https://www.vishvasnews.com"
                   target="_blank">✅ Vishvas News</a>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:11px;
                        color:rgba(255,255,255,0.2);
                        margin-top:16px;line-height:1.6">
                Disclaimer: AI prediction only.
                Always verify from official sources.
            </div>
            """, unsafe_allow_html=True)

    elif not analyze_btn:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:44px;margin-bottom:14px">🔍</div>
            <div style="font-size:15px;
                        color:rgba(255,255,255,0.2);
                        line-height:1.8">
                Paste a WhatsApp forward on the left<br>
                and click Analyze
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">🔍 VeriLang</div>
    <div style="font-size:11px;color:rgba(255,255,255,0.25);
                margin-bottom:2px">
        Vernacular Truth Detector v2.0
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;letter-spacing:2px;
                text-transform:uppercase;
                color:rgba(255,255,255,0.2);
                margin-bottom:10px">
        Accuracy per Language
    </div>
    """, unsafe_allow_html=True)

    for lang, acc in {
        "Hindi":99.80,"Gujarati":99.93,
        "Marathi":100.0,"Telugu":100.0
    }.items():
        st.markdown(f"""
        <div class="acc-row">
            <span class="acc-lang">{lang}</span>
            <span class="acc-val">{acc}%</span>
        </div>
        <div class="acc-track">
            <div class="acc-fill" style="width:{acc}%"></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;letter-spacing:2px;
                text-transform:uppercase;
                color:rgba(255,255,255,0.2);
                margin-bottom:10px">
        Model
    </div>
    <div style="font-size:12px;color:rgba(255,255,255,0.35);
                line-height:2">
        google/muril-base-cased<br>
        Fine-tuned on 49,426 articles<br>
        + Hinglish augmentation<br>
        PyTorch · HuggingFace · SHAP
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;letter-spacing:2px;
                text-transform:uppercase;
                color:rgba(255,255,255,0.2);
                margin-bottom:10px">
        Dataset
    </div>
    <div style="font-size:11px;color:rgba(255,255,255,0.25);
                line-height:1.8">
        Zenodo Multilingual Fake News<br>
        Hindi · Gujarati · Marathi · Telugu
    </div>
    """, unsafe_allow_html=True)

    try:
        st.image(
            'shap_all_languages.png',
            caption='SHAP word importance',
            use_column_width=True
        )
    except Exception:
        pass