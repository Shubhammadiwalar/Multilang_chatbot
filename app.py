import os, hashlib, io
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from gtts import gTTS
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import pydub, pydub.utils

_FF  = r"C:\Users\shubh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"
_FFP = r"C:\Users\shubh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"
pydub.AudioSegment.converter = _FF
pydub.utils.get_encoder_name = lambda: _FF
pydub.utils.get_prober_name  = lambda: _FFP
from pydub import AudioSegment

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="NOVA AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# session state
for k, v in {
    "screen": "landing", "messages": [], "file_hash": None,
    "vector_store": None, "document_name": None,
    "chunk_count": 0, "tts_enabled": True,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:#03040a; --panel:#070c1a; --glass:rgba(7,12,26,0.88);
    --b:rgba(0,212,255,0.12); --bhi:rgba(0,212,255,0.42);
    --cyan:#00d4ff; --violet:#7c3aed; --pink:#f000b8;
    --green:#00ff88; --text:#d8e8ff; --muted:#3a4f6a;
}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stAppViewContainer"]{background:transparent!important;}
[data-testid="stSidebar"]{display:none!important;}
section[data-testid="stSidebarContent"]{display:none!important;}
[data-testid="stSidebarCollapsedControl"]{display:none!important;}

.stApp{
    background:
        linear-gradient(rgba(0,212,255,0.018) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,212,255,0.018) 1px,transparent 1px),
        radial-gradient(ellipse at 18% 0%,rgba(124,58,237,0.22) 0%,transparent 50%),
        radial-gradient(ellipse at 82% 8%,rgba(0,212,255,0.15) 0%,transparent 45%),
        radial-gradient(ellipse at 50% 100%,rgba(240,0,184,0.10) 0%,transparent 42%),
        var(--bg)!important;
    background-size:52px 52px,52px 52px,100% 100%,100% 100%,100% 100%,100% 100%!important;
    color:var(--text)!important;
}
.block-container{max-width:100%!important;padding:0!important;}
::-webkit-scrollbar{width:3px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:rgba(0,212,255,0.18);border-radius:2px;}

@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes hexGlow{
    0%,100%{filter:drop-shadow(0 0 16px rgba(0,212,255,0.55)) drop-shadow(0 0 32px rgba(124,58,237,0.3));}
    50%{filter:drop-shadow(0 0 32px rgba(0,212,255,0.9)) drop-shadow(0 0 64px rgba(124,58,237,0.55));}
}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes msgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ── LANDING ── */
.land-page{text-align:center;padding:60px 24px 24px;display:flex;flex-direction:column;align-items:center;min-height:82vh;}
.logo-orbit{position:relative;width:148px;height:148px;margin:0 auto 40px;}
.or1{position:absolute;inset:-20px;border-radius:50%;border:1px solid rgba(0,212,255,0.18);animation:spin 9s linear infinite;}
.or1::before{content:'';position:absolute;top:-5px;left:50%;transform:translateX(-50%);width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 14px var(--cyan),0 0 28px var(--cyan);}
.or2{position:absolute;inset:-42px;border-radius:50%;border:1px solid rgba(124,58,237,0.14);animation:spin 16s linear infinite reverse;}
.or2::before{content:'';position:absolute;bottom:-5px;left:50%;transform:translateX(-50%);width:7px;height:7px;border-radius:50%;background:var(--violet);box-shadow:0 0 12px var(--violet),0 0 24px var(--violet);}
.logo-svg{width:148px;height:148px;display:flex;align-items:center;justify-content:center;position:relative;z-index:2;animation:hexGlow 3.5s ease-in-out infinite;}
.land-tag{display:inline-flex;align-items:center;gap:8px;padding:5px 15px;border-radius:4px;border:1px solid rgba(0,212,255,0.26);background:rgba(0,212,255,0.055);color:var(--cyan);font-size:9px;font-weight:700;letter-spacing:3px;font-family:'JetBrains Mono',monospace;margin-bottom:20px;}
.land-tag-dot{width:5px;height:5px;border-radius:50%;background:var(--cyan);box-shadow:0 0 6px var(--cyan);animation:blink 2s ease-in-out infinite;}
.land-title{font-size:clamp(58px,9vw,108px);font-weight:800;letter-spacing:-4px;line-height:1;color:#fff;margin:0 0 8px;}
.land-grad{background:linear-gradient(90deg,var(--cyan) 0%,var(--violet) 45%,var(--pink) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.land-sub{font-size:clamp(14px,1.8vw,18px);color:#4a6080;line-height:1.8;max-width:520px;margin:0 auto 44px;}
.land-pills{display:flex;flex-wrap:wrap;justify-content:center;gap:9px;margin-bottom:44px;}
.lpill{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:6px;border:1px solid var(--b);background:rgba(0,212,255,0.035);color:#607898;font-size:11px;font-weight:500;font-family:'JetBrains Mono',monospace;}
.lpill-dot{width:4px;height:4px;border-radius:50%;background:var(--cyan);opacity:0.6;}
.land-stats{display:flex;gap:44px;justify-content:center;margin-top:44px;padding-top:28px;border-top:1px solid var(--b);flex-wrap:wrap;}
.lsn{font-size:22px;font-weight:700;color:var(--cyan);font-family:'JetBrains Mono',monospace;text-align:center;}
.lsl{font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:3px;text-transform:uppercase;text-align:center;}
.land-foot{margin-top:36px;color:var(--muted);font-size:9px;letter-spacing:2px;font-family:'JetBrains Mono',monospace;}

/* CTA */
.cta-btn button{
    background:linear-gradient(135deg,var(--cyan),var(--violet))!important;
    border:none!important;border-radius:10px!important;color:#fff!important;
    font-size:14px!important;font-weight:700!important;letter-spacing:2px!important;
    padding:14px 0!important;
    box-shadow:0 0 32px rgba(0,212,255,0.32),0 0 64px rgba(124,58,237,0.20)!important;
    transition:all 0.25s ease!important;
}
.cta-btn button:hover{transform:translateY(-2px)!important;box-shadow:0 0 52px rgba(0,212,255,0.6),0 0 104px rgba(124,58,237,0.38)!important;}

/* ── APP LEFT PANEL — style via column container ── */
div[data-testid="stHorizontalBlock"] > div:first-child > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] > div:first-child > div > div[data-testid="stVerticalBlock"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--b) !important;
    min-height: 100vh !important;
    padding: 20px 18px !important;
    position: relative;
}
/* top accent line on left panel */
div[data-testid="stHorizontalBlock"] > div:first-child {
    background: var(--panel) !important;
    border-right: 1px solid var(--b) !important;
}
.sb-brand{display:flex;align-items:center;gap:12px;padding-bottom:18px;border-bottom:1px solid var(--b);margin-bottom:14px;}
.sb-hex{width:38px;height:38px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--cyan),var(--violet));clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);box-shadow:0 0 18px rgba(0,212,255,0.35);}
.sb-name{font-size:17px;font-weight:700;color:#fff;letter-spacing:2px;}
.sb-sub{font-size:9px;color:var(--cyan);letter-spacing:2.5px;font-family:'JetBrains Mono',monospace;}
.sb-label{font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--muted);margin:14px 0 8px;display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;}
.sb-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--b),transparent);}
.online-pill{display:inline-flex;align-items:center;gap:7px;padding:6px 12px;border-radius:5px;border:1px solid rgba(0,255,136,0.20);background:rgba(0,255,136,0.05);color:var(--green);font-size:9px;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:1px;margin-bottom:12px;}
.online-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 2s ease-in-out infinite;}
.doc-chip{background:linear-gradient(135deg,rgba(0,212,255,0.055),rgba(124,58,237,0.055));border:1px solid var(--bhi);border-radius:9px;padding:11px 13px;position:relative;overflow:hidden;margin-bottom:8px;}
.doc-chip::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);}
.dc-name{font-size:11px;font-weight:600;color:#b8d4ff;word-break:break-all;}
.dc-meta{font-size:9px;color:var(--muted);margin-top:4px;font-family:'JetBrains Mono',monospace;}
.drop-zone{border:1px dashed rgba(0,212,255,0.24);border-radius:12px;background:rgba(0,212,255,0.018);padding:6px;}
.feat-list{display:flex;flex-direction:column;gap:7px;margin-top:4px;}
.fi{display:flex;align-items:flex-start;gap:9px;padding:9px 11px;border-radius:8px;border:1px solid var(--b);background:rgba(4,8,18,0.55);}
.fi-ico{font-size:13px;flex-shrink:0;}
.fi-t{font-size:10px;font-weight:600;color:#a8c4e8;margin-bottom:1px;}
.fi-d{font-size:9px;color:var(--muted);line-height:1.5;}
.sb-foot{color:var(--muted);font-size:9px;line-height:2;font-family:'JetBrains Mono',monospace;padding:10px 0 2px;border-top:1px solid var(--b);margin-top:14px;}

/* ── APP RIGHT PANEL ── */
.topbar{display:flex;align-items:center;justify-content:space-between;padding-bottom:16px;border-bottom:1px solid var(--b);margin-bottom:18px;}
.tb-left{display:flex;align-items:center;gap:10px;}
.tb-title{font-size:15px;font-weight:700;color:#fff;letter-spacing:1px;}
.tb-sub{font-size:9px;color:var(--cyan);letter-spacing:2.5px;font-family:'JetBrains Mono',monospace;}
.tb-status{display:flex;align-items:center;gap:6px;font-size:9px;font-family:'JetBrains Mono',monospace;color:var(--green);letter-spacing:1px;}
.tb-dot{width:6px;height:6px;background:var(--green);border-radius:50%;box-shadow:0 0 8px var(--green);animation:blink 2s ease-in-out infinite;}

.welcome-card{background:linear-gradient(135deg,rgba(0,212,255,0.045) 0%,rgba(124,58,237,0.07) 60%,rgba(240,0,184,0.035) 100%);border:1px solid var(--b);border-radius:14px;padding:28px;position:relative;overflow:hidden;margin-bottom:24px;}
.welcome-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),var(--violet),transparent);}
.wc-scan{position:absolute;inset:0;border-radius:14px;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,212,255,0.008) 3px,rgba(0,212,255,0.008) 4px);}
.wc-title{font-size:22px;font-weight:700;color:#fff;letter-spacing:-0.5px;margin-bottom:8px;}
.wc-desc{font-size:13px;color:#4a6080;line-height:1.75;max-width:520px;}
.wc-examples{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;}
.wce{padding:7px 14px;border-radius:6px;border:1px solid var(--b);background:rgba(0,212,255,0.03);color:#5a7498;font-size:11px;font-style:italic;font-family:'JetBrains Mono',monospace;}

.ready-pill{display:inline-flex;align-items:center;gap:8px;padding:7px 14px;border-radius:6px;border:1px solid rgba(0,255,136,0.18);background:rgba(0,255,136,0.04);color:var(--green);font-size:9px;font-weight:700;font-family:'JetBrains Mono',monospace;letter-spacing:1px;margin-bottom:16px;}
.rp-dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 2s ease-in-out infinite;}

[data-testid="stChatMessage"]{background:var(--glass)!important;border:1px solid var(--b)!important;border-radius:12px!important;padding:16px 20px!important;margin-bottom:10px!important;backdrop-filter:blur(14px)!important;animation:msgIn 0.3s ease both!important;}
[data-testid="stChatMessageContent"]{color:#c0d4f0!important;font-size:14px!important;line-height:1.9!important;}
[data-testid="stChatInput"]{background:rgba(4,7,18,0.95)!important;border:1px solid var(--bhi)!important;border-radius:12px!important;box-shadow:0 0 24px rgba(0,212,255,0.10)!important;}
[data-testid="stChatInput"]:focus-within{border-color:var(--cyan)!important;box-shadow:0 0 36px rgba(0,212,255,0.20)!important;}
[data-testid="stChatInput"] textarea{color:#fff!important;font-size:14px!important;}

iframe[title="streamlit_mic_recorder"]{height:46px!important;width:46px!important;border-radius:50%!important;border:1.5px solid rgba(0,212,255,0.45)!important;background:rgba(0,212,255,0.07)!important;box-shadow:0 0 18px rgba(0,212,255,0.20)!important;display:block!important;}

[data-testid="stFileUploader"]{background:transparent!important;border:none!important;}
[data-testid="stExpander"]{background:rgba(4,7,18,0.6)!important;border:1px solid var(--b)!important;border-radius:8px!important;}
.stSpinner>div{border-top-color:var(--cyan)!important;}
.stButton>button{border-radius:7px!important;border:1px solid var(--bhi)!important;background:rgba(0,212,255,0.05)!important;color:var(--cyan)!important;font-weight:600!important;font-size:11px!important;letter-spacing:0.8px!important;transition:all 0.2s ease!important;}
.stButton>button:hover{background:rgba(0,212,255,0.11)!important;box-shadow:0 0 16px rgba(0,212,255,0.18)!important;transform:translateY(-1px)!important;}
[data-testid="stSelectbox"]>div>div{background:rgba(5,9,20,0.92)!important;border:1px solid var(--b)!important;border-radius:7px!important;color:var(--text)!important;}
.voice-result{background:rgba(0,212,255,0.055);border:1px solid rgba(0,212,255,0.28);border-radius:8px;padding:10px 14px;margin:6px 0;font-size:13px;color:#c0d4f0;}
.app-footer{text-align:center;color:var(--muted);font-size:9px;letter-spacing:2px;font-family:'JetBrains Mono',monospace;padding:20px 0 2px;border-top:1px solid var(--b);margin-top:32px;}
</style>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()

# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════
@st.cache_resource
def get_models():
    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=API_KEY, temperature=0.2, streaming=True)
    return emb, llm

def extract_text(f):
    docs = []
    if f.name.lower().endswith(".pdf"):
        for i, page in enumerate(PdfReader(f).pages, 1):
            t = page.extract_text() or ""
            if t.strip():
                docs.append(Document(page_content=t, metadata={"source": f.name, "page": i}))
    elif f.name.lower().endswith(".txt"):
        t = f.getvalue().decode("utf-8-sig", errors="replace")
        if t.strip():
            docs.append(Document(page_content=t, metadata={"source": f.name, "page": None}))
    return docs

def build_index(docs, emb):
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    if not chunks:
        raise ValueError("No readable text found.")
    return FAISS.from_documents(chunks, emb), len(chunks)

def token_stream(llm, prompt):
    for chunk in llm.stream(prompt):
        c = chunk.content
        if not c:
            continue
        if isinstance(c, list):
            for p in c:
                yield p.get("text", "") if isinstance(p, dict) else str(p)
        else:
            yield str(c)

GTTS = {"Auto-detect":"en","English":"en","Hindi":"hi","Kannada":"kn","Tamil":"ta","Telugu":"te","Marathi":"mr","Bengali":"bn"}
STTL = {"Auto-detect":"en-US","English":"en-US","Hindi":"hi-IN","Kannada":"kn-IN","Tamil":"ta-IN","Telugu":"te-IN","Marathi":"mr-IN","Bengali":"bn-IN"}

def transcribe(audio_bytes, lang="Auto-detect"):
    rec = sr.Recognizer()
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav = io.BytesIO()
        seg.export(wav, format="wav")
        wav.seek(0)
        with sr.AudioFile(wav) as src:
            aud = rec.record(src)
        return rec.recognize_google(aud, language=STTL.get(lang, "en-US"))
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        return f"[STT error: {e}]"

BIG_SVG = '<svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="lg" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#00d4ff"/><stop offset="50%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#f000b8"/></linearGradient><filter id="gf"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><polygon points="50,7 90,28 90,72 50,93 10,72 10,28" fill="none" stroke="url(#lg)" stroke-width="1.5" opacity="0.75"/><polygon points="50,18 76,33 76,67 50,82 24,67 24,33" fill="none" stroke="url(#lg)" stroke-width="1" opacity="0.32"/><circle cx="50" cy="50" r="17" fill="none" stroke="url(#lg)" stroke-width="1.5" opacity="0.9"/><circle cx="50" cy="50" r="7" fill="url(#lg)" filter="url(#gf)"/><line x1="50" y1="33" x2="50" y2="7" stroke="#00d4ff" stroke-width="0.8" opacity="0.6"/><line x1="50" y1="67" x2="50" y2="93" stroke="#f000b8" stroke-width="0.8" opacity="0.6"/><line x1="35" y1="41" x2="10" y2="28" stroke="#00d4ff" stroke-width="0.8" opacity="0.5"/><line x1="65" y1="41" x2="90" y2="28" stroke="#7c3aed" stroke-width="0.8" opacity="0.5"/><line x1="35" y1="59" x2="10" y2="72" stroke="#7c3aed" stroke-width="0.8" opacity="0.5"/><line x1="65" y1="59" x2="90" y2="72" stroke="#f000b8" stroke-width="0.8" opacity="0.5"/><circle cx="50" cy="7" r="3.5" fill="#00d4ff" filter="url(#gf)"/><circle cx="90" cy="28" r="3" fill="#7c3aed" filter="url(#gf)"/><circle cx="90" cy="72" r="3" fill="#f000b8" filter="url(#gf)"/><circle cx="50" cy="93" r="3.5" fill="#00d4ff" filter="url(#gf)"/><circle cx="10" cy="72" r="3" fill="#7c3aed" filter="url(#gf)"/><circle cx="10" cy="28" r="3" fill="#f000b8" filter="url(#gf)"/></svg>'
TINY_SVG = '<svg width="22" height="22" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="tg" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#00d4ff"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient></defs><polygon points="50,7 90,28 90,72 50,93 10,72 10,28" fill="none" stroke="url(#tg)" stroke-width="8"/><circle cx="50" cy="50" r="18" fill="url(#tg)"/></svg>'

# ════════════════════════════════════════════════════════
#  SCREEN 1 — LANDING
# ════════════════════════════════════════════════════════
if st.session_state.screen == "landing":

    st.markdown(f"""
    <div class="land-page">
        <div class="logo-orbit">
            <div class="or1"></div>
            <div class="or2"></div>
            <div class="logo-svg">{BIG_SVG}</div>
        </div>
        <div class="land-tag"><span class="land-tag-dot"></span>NEURAL DOCUMENT INTELLIGENCE</div>
        <div class="land-title"><span class="land-grad">NOVA</span> AI</div>
        <p class="land-sub">Upload any document. Ask questions in your language.<br>Get instant streaming answers powered by Gemini AI.</p>
        <div class="land-pills">
            <span class="lpill"><span class="lpill-dot"></span>PDF &amp; TXT</span>
            <span class="lpill"><span class="lpill-dot"></span>8+ Languages</span>
            <span class="lpill"><span class="lpill-dot"></span>Voice I/O</span>
            <span class="lpill"><span class="lpill-dot"></span>RAG &amp; FAISS</span>
            <span class="lpill"><span class="lpill-dot"></span>Streaming</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, c2, _ = st.columns([2, 1.6, 2])
    with c2:
        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        if st.button("⬡  GET STARTED  →", use_container_width=True, key="cta"):
            st.session_state.screen = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="land-stats">
        <div><div class="lsn">RAG</div><div class="lsl">Vector Search</div></div>
        <div><div class="lsn">8+</div><div class="lsl">Languages</div></div>
        <div><div class="lsn">🎙</div><div class="lsl">Voice I/O</div></div>
        <div><div class="lsn">⚡</div><div class="lsl">Streaming</div></div>
    </div>
    <div class="land-foot">NOVA AI · v2.0 · GEMINI + LANGCHAIN + FAISS</div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
#  SCREEN 2 — APP  (two-column layout, NO sidebar)
# ════════════════════════════════════════════════════════
elif st.session_state.screen == "app":

    left, right = st.columns([1, 2.8], gap="small")

    # ── LEFT PANEL ───────────────────────────────────────
    with left:
        st.markdown(f"""
        <div class="sb-brand">
            <div class="sb-hex">{TINY_SVG}</div>
            <div><div class="sb-name">NOVA AI</div><div class="sb-sub">DOC INTELLIGENCE</div></div>
        </div>
        <div class="online-pill"><span class="online-dot"></span>GEMINI · ONLINE</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Upload Document</div>', unsafe_allow_html=True)
        st.markdown('<div class="drop-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("file", type=["pdf", "txt"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            h = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            if h != st.session_state.file_hash:
                st.session_state.update({
                    "file_hash": h, "vector_store": None,
                    "messages": [], "document_name": None, "chunk_count": 0,
                })
                try:
                    with st.spinner("⬡ Building index…"):
                        uploaded_file.seek(0)
                        docs = extract_text(uploaded_file)
                        if not docs:
                            st.warning("No readable text found.")
                            st.stop()
                        emb, _ = get_models()
                        vs, cnt = build_index(docs, emb)
                        st.session_state.update({
                            "vector_store": vs,
                            "document_name": uploaded_file.name,
                            "chunk_count": cnt,
                        })
                except Exception as e:
                    st.session_state.file_hash = None
                    st.error("Processing failed.")
                    with st.expander("Error"):
                        st.code(str(e))
                    st.stop()

            st.markdown(f"""
            <div class="doc-chip">
                <div class="dc-name">📄 {st.session_state.document_name}</div>
                <div class="dc-meta">▸ {st.session_state.chunk_count} chunks · indexed</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sb-label">Language</div>', unsafe_allow_html=True)
        language = st.selectbox("lang",
            ["Auto-detect","English","Hindi","Kannada","Tamil","Telugu","Marathi","Bengali"],
            label_visibility="collapsed")

        st.markdown('<div class="sb-label">Voice</div>', unsafe_allow_html=True)
        st.session_state.tts_enabled = st.toggle("🔊 Read answers aloud", value=st.session_state.tts_enabled)

        st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)
        if st.button("⟳  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("⌂  Home", use_container_width=True):
            st.session_state.screen = "landing"
            st.session_state.messages = []
            st.rerun()

        st.markdown("""
        <div class="feat-list">
            <div class="fi"><div class="fi-ico">⬡</div><div><div class="fi-t">Vector Retrieval</div><div class="fi-d">FAISS semantic search.</div></div></div>
            <div class="fi"><div class="fi-ico">🎙</div><div><div class="fi-t">Voice I/O</div><div class="fi-d">Mic input + TTS output.</div></div></div>
            <div class="fi"><div class="fi-ico">◉</div><div><div class="fi-t">Multilingual</div><div class="fi-d">8+ languages supported.</div></div></div>
        </div>
        <div class="sb-foot">NOVA AI v2.0 · Gemini · LangChain · FAISS</div>
        """, unsafe_allow_html=True)

    # ── RIGHT PANEL ──────────────────────────────────────
    with right:
        st.markdown("""
        <div class="topbar">
            <div class="tb-left">
                <span style="font-size:20px;">⬡</span>
                <div><div class="tb-title">NOVA AI Chat</div><div class="tb-sub">DOCUMENT INTELLIGENCE</div></div>
            </div>
            <div class="tb-status"><span class="tb-dot"></span>ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.vector_store:
            st.markdown(f"""
            <div class="welcome-card">
                <div class="wc-scan"></div>
                <div class="wc-title">Welcome to <span class="land-grad">NOVA AI</span></div>
                <p class="wc-desc">Upload a PDF or TXT from the left panel. Once indexed, ask anything in any language and get streaming answers.</p>
                <div class="wc-examples">
                    <span class="wce">▸ Summarise the document</span>
                    <span class="wce">▸ What are the key findings?</span>
                    <span class="wce">▸ इस दस्तावेज़ का विषय क्या है?</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ready-pill">
                <span class="rp-dot"></span>
                INDEXED · READY — {st.session_state.chunk_count} chunks · {st.session_state.document_name}
            </div>
            """, unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("◈ Sources"):
                            for src in msg["sources"]:
                                lbl = f"{src['source']} · Page {src['page']}" if src["page"] else src["source"]
                                st.markdown(f"**`{lbl}`**")
                                st.caption(src["content"][:600])

            # mic + input
            c_mic, c_inp = st.columns([1, 14])
            with c_mic:
                audio = mic_recorder(start_prompt="🎙", stop_prompt="⏹", just_once=True, use_container_width=True, key="mic")
            with c_inp:
                question = st.chat_input("Ask anything about your document…")

            if audio and audio.get("bytes"):
                with st.spinner("🎙 Transcribing…"):
                    vt = transcribe(audio["bytes"], language)
                if vt and not vt.startswith("[STT error"):
                    question = vt
                    st.success(f'🎙 Heard: "{vt}"')
                elif vt.startswith("[STT error"):
                    st.error(vt)
                else:
                    st.warning("Could not understand audio.")

            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    try:
                        _, llm = get_models()
                        fq = f"Answer in {language}. Question: {question}" if language != "Auto-detect" else question

                        with st.spinner("⬡ Searching…"):
                            rdocs = st.session_state.vector_store.similarity_search(fq, k=3)

                        if not rdocs:
                            answer = "I couldn't find relevant information."
                            sources = []
                            st.markdown(answer)
                        else:
                            ctx = "\n\n".join(
                                f"Source: {d.metadata.get('source','?')}, Page: {d.metadata.get('page','?')}\n{d.page_content[:800]}"
                                for d in rdocs
                            )
                            prompt = f"""You are NOVA AI, a multilingual document assistant.
Answer using only the context. Reply in the same language as the question.
Never invent facts. Say so if context is insufficient.

CONTEXT:
{ctx}

QUESTION:
{fq}

ANSWER:"""
                            answer = st.write_stream(token_stream(llm, prompt))
                            sources = [{"source": d.metadata.get("source","?"), "page": d.metadata.get("page"), "content": d.page_content} for d in rdocs]

                        if st.session_state.get("tts_enabled") and answer:
                            try:
                                buf = io.BytesIO()
                                gTTS(text=answer[:3000], lang=GTTS.get(language, "en"), slow=False).write_to_fp(buf)
                                buf.seek(0)
                                st.audio(buf, format="audio/mp3")
                            except Exception:
                                pass

                        if sources:
                            with st.expander("◈ Sources"):
                                for src in sources:
                                    lbl = f"{src['source']} · Page {src['page']}" if src["page"] else src["source"]
                                    st.markdown(f"**`{lbl}`**")
                                    st.caption(src["content"][:600])

                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

                    except Exception as e:
                        st.error("Generation failed.")
                        with st.expander("Error"):
                            st.code(str(e))

        st.markdown('<div class="app-footer">NOVA AI · Answers from document context only · Verify critical information</div>', unsafe_allow_html=True)
