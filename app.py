import streamlit as st
import re
import csv
import uuid
import unicodedata
from io import StringIO, BytesIO
from pathlib import Path
from collections import Counter
import pythainlp
from pythainlp.tokenize import word_tokenize

# Try to import optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ไทยอักษร - Thai Text Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --ink: #1a1410;
        --paper: #f7f3ee;
        --gold: #c8a96e;
        --rust: #c05a2a;
        --teal: #2a7a7a;
    }
    .main {
        background-color: var(--paper);
    }
    h1, h2, h3 {
        color: var(--ink);
    }
    .stButton>button {
        background-color: var(--rust);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #a04020;
    }
</style>
""", unsafe_allow_html=True)

# ─── Thai Text Normalization ──────────────────────────────────────────────────

def normalize_thai_text(text: str) -> str:
    """
    Normalize Thai text:
    - Fix NFC unicode normalization
    - Remove stray/duplicate diacritics
    - Fix common OCR misreads
    """
    if not text:
        return text

    # NFC normalize
    text = unicodedata.normalize('NFC', text)

    # Fix common Thai misreads
    ocr_fixes = {
        'ํา': 'ำ',
        '\u0e4d\u0e32': '\u0e33',
        'เแ': 'แ',
        'เแ': 'แ',
        '่่': '่',
        '้้': '้',
        '๊๊': '๊',
        '๋๋': '๋',
        '็็': '็',
    }
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)

    # Remove zero-width chars
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    text = text.replace('\ufeff', '')

    # Collapse multiple spaces/newlines
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def split_into_segments(text: str) -> list:
    """Split normalized Thai text into translation segments."""
    pattern = re.compile(
        r'(?<=[ๆ\.\!\?\u0e2f\u0e5a\u0e5b])\s+'
        r'|(?<=\n)'
        r'|\n'
    )
    raw = pattern.split(text)
    segments = []
    for seg in raw:
        seg = seg.strip()
        if seg and len(seg) > 1:
            segments.append(seg)
    return segments


def extract_glossary(segments: list, top_n: int = 50, min_len: int = 2, min_freq: int = 2) -> list:
    """Extract frequently repeated Thai words/phrases."""
    thai_char = re.compile(r'[\u0e00-\u0e7f]')
    all_tokens = []
    
    for seg in segments:
        tokens = word_tokenize(seg, engine='newmm', keep_whitespace=False)
        for tok in tokens:
            tok = tok.strip()
            if (len(tok) >= min_len
                    and thai_char.search(tok)
                    and tok not in ('ๆ', 'ฯ', 'และ', 'หรือ', 'ใน', 'ที่', 'ของ', 'การ',
                                     'ให้', 'เป็น', 'ได้', 'จาก', 'โดย', 'มี', 'กับ')):
                all_tokens.append(tok)

    freq = Counter(all_tokens)
    glossary = [
        {"term": term, "frequency": count, "translation": ""}
        for term, count in freq.most_common(top_n)
        if count >= min_freq
    ]
    return glossary


def extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF - text layer only (no OCR)."""
    if not PYMUPDF_AVAILABLE:
        st.error("PyMuPDF not installed. Install: pip install PyMuPDF")
        return ""

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text_parts = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            full_text_parts.append(text)

    doc.close()
    return "\n\n".join(full_text_parts)


def segments_to_csv(segments: list, filename: str) -> str:
    """Generate CAT-tool-ready CSV from segments."""
    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["ID", "Source (TH)", "Target", "Notes"])
    for i, seg in enumerate(segments, 1):
        writer.writerow([f"{filename}_{i:04d}", seg, "", ""])
    return output.getvalue()


def glossary_to_csv(glossary: list) -> str:
    """Generate glossary CSV."""
    output = StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(["Term (TH)", "Frequency", "Translation", "Notes"])
    for item in glossary:
        writer.writerow([item["term"], item["frequency"], item["translation"], ""])
    return output.getvalue()


# ─── Streamlit App ───────────────────────────────────────────────────────────

st.markdown("""
# 📄 ไทยอักษร · Thai Text Extractor

สกัดข้อความภาษาไทยจาก PDF พร้อมสร้าง Glossary สำหรับนักแปล
""")

st.info("✨ ระบบสกัดข้อความภาษาไทยแบบอัจฉริยะ สำหรับ CAT Tools")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่า")
    top_n_glossary = st.slider("จำนวนคำในคลังศัพท์", 10, 100, 50)
    min_frequency = st.slider("ความถี่ต่ำสุด", 1, 10, 2)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 อัปโหลดไฟล์")
    st.markdown("> รองรับ: **PDF** (ต้องเป็น text-based ไม่ใช่ scanned)")
    
    uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type=["pdf"])

with col2:
    st.markdown("### ℹ️ วิธีใช้")
    st.markdown("""
    1. อัปโหลดไฟล์ PDF ที่มีข้อความ
    2. ระบบจะสกัดข้อความและแบ่งเป็น segments
    3. สกัดคำที่พบซ้ำบ่อยเป็น glossary
    4. ดาวน์โหลด CSV ไปใน CAT Tools
    """)

# Processing
if uploaded_file is not None:
    st.markdown("---")
    st.markdown("### 🔄 กำลังประมวลผล...")
    
    # Read file
    file_bytes = uploaded_file.read()
    filename = Path(uploaded_file.name).stem
    
    try:
        # Extract text
        progress_bar = st.progress(0)
        st.info("📖 สกัดข้อความจาก PDF...")
        
        raw_text = extract_from_pdf(file_bytes)
        progress_bar.progress(20)
        
        if not raw_text.strip():
            st.error("❌ ไม่พบข้อความในไฟล์ PDF กรุณาใช้ PDF ที่เป็น text-based (ไม่ใช่ scanned)")
            st.stop()
        
        # Normalize
        st.info("🔤 ทำให้ข้อความเป็นมาตรฐาน...")
        normalized = normalize_thai_text(raw_text)
        progress_bar.progress(40)
        
        # Split segments
        st.info("✂️ แบ่งประโยค...")
        segments = split_into_segments(normalized)
        progress_bar.progress(60)
        
        if not segments:
            st.error("❌ ไม่พบข้อความภาษาไทยหลังการประมวลผล")
            st.stop()
        
        # Extract glossary
        st.info("📚 สกัดคลังศัพท์...")
        glossary = extract_glossary(segments, top_n=top_n_glossary, min_freq=min_frequency)
        progress_bar.progress(100)
        
        st.success("✅ สำเร็จ!")
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 ผลลัพธ์")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Segments", len(segments))
        with col2:
            st.metric("📚 คำศัพท์", len(glossary))
        with col3:
            st.metric("🔤 ตัวอักษร", len(normalized))
        with col4:
            st.metric("📄 ไฟล์", uploaded_file.name)
        
        # Raw preview
        with st.expander("👁️ ดูข้อความเต็ม (2000 ตัวอักษร)"):
            st.text(normalized[:2000])
        
        # Segments preview
        st.markdown("### 📖 Segments Preview (100 รายการแรก)")
        
        seg_df_data = {
            "ID": [f"{filename}_{i:04d}" for i in range(1, min(101, len(segments) + 1))],
            "ข้อความ": segments[:100],
        }
        
        st.dataframe(seg_df_data, use_container_width=True, height=400)
        
        # Glossary table
        st.markdown("### 📚 Glossary (คำซ้ำบ่อย)")
        
        glo_df_data = {
            "คำศัพท์": [g["term"] for g in glossary],
            "ความถี่": [g["frequency"] for g in glossary],
            "คำแปล": ["" for _ in glossary],
        }
        
        st.dataframe(glo_df_data, use_container_width=True, height=400)
        
        # Download buttons
        st.markdown("---")
        st.markdown("### ⬇️ ดาวน์โหลด")
        
        col1, col2 = st.columns(2)
        
        with col1:
            seg_csv = segments_to_csv(segments, filename)
            st.download_button(
                label="📥 ดาวน์โหลด segments.csv (CAT Import)",
                data=seg_csv.encode('utf-8-sig'),
                file_name="segments.csv",
                mime="text/csv",
                key="download_segments"
            )
        
        with col2:
            glo_csv = glossary_to_csv(glossary)
            st.download_button(
                label="📥 ดาวน์โหลด glossary.csv (Term Base)",
                data=glo_csv.encode('utf-8-sig'),
                file_name="glossary.csv",
                mime="text/csv",
                key="download_glossary"
            )
        
        # Statistics
        with st.expander("📈 สถิติเพิ่มเติม"):
            st.markdown(f"""
            - **จำนวน segments:** {len(segments)}
            - **จำนวนคำศัพท์:** {len(glossary)}
            - **ตัวอักษรทั้งหมด:** {len(normalized)}
            - **เฉลี่ยตัวอักษร/segment:** {len(normalized) / len(segments):.1f}
            - **คำศัพท์ที่พบบ่อยสุด:** {glossary[0]['term'] if glossary else 'N/A'} ({glossary[0]['frequency'] if glossary else 'N/A'} ครั้ง)
            """)
    
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.info("💡 แนวทาง: ตรวจสอบว่า PDF เป็น text-based (ไม่ใช่ scanned)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7a6e60; font-size: 12px;">
    <p>ไทยอักษร · Thai Text Extractor v1.0</p>
    <p>Powered by PyThaiNLP · PyMuPDF · Streamlit</p>
    <p><a href="https://github.com/hafmyclothes/thai-text-extractor">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
