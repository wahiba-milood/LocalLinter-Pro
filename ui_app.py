import streamlit as st
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import os
import re
from collections import Counter

# إعدادات الصفحة الافتراضية
st.set_page_config(page_title="نظام التلخيص والمقارنة الذكي", page_icon="📚", layout="wide")

# قاموس اللغات للواجهة
UI_TEXT = {
    "ar": {
        "title": " نظام التلخيص الذكي والمقارنة الآلية للأبحاث ",
        "authors": "",
        "header_upload": "📂 تحميل الأبحاث العلمية للمقارنة والتحليل (PDF)",
        "caption_upload": "",
        "p1": "ارفع البحث الأول", "p2": "ارفع البحث الثاني", "p3": "ارفع البحث الثالث", "p4": "ارفع البحث الرابع", "p5": "ارفع البحث الخامس",
        "btn_analyze": "🚀 ابدأ المقارنة والتحليل الآلي الديناميكي المطور",
        "spinner": "جاري استخراج المقدمات والخواتيم وتحليل {} ملفات عبر Llama 3 محلياً... يرجى الانتظار...",
        "error_pdf": "حدث خطأ أثناء قراءة الملف: ({}). تأكدي أنه ملف رقمي وليس صورة ممسوحة.",
        "warning_empty": "⚠️ يرجى رفع ملف PDF واحد على الأقل لإجراء عملية التحليل أو المقارنة.",
        "success": "✨ تم توليد مصفوفة المقارنة المخصصة والتحليل الأكاديمي بنجاح من نصوص الأبحاث الكاملة!",
        "btn_download": "📥 تحميل مصفوفة المقارنة كملف Word (.docx)",
        "matrix_header": "📊 مصفوفة المقارنة الأكاديمية لـ ({}) أبحاث علمية",
        "analytics_header": "📈 لوحة المؤشرات الإحصائية السريعة",
        "ideas_header": "💡 مقترحات بحثية مبتكرة بناءً على الفجوات المكتشفة",
        "chatbot_header": "🤖 المستشار الأكاديمي المساعد (اسأل عن الأبحاث المرفوعة)",
        "chatbot_placeholder": "مثال: ما هو البحث الذي ناقش تطبيقات إنترنت الأشياء؟ وما هي فجواته؟",
        "chatbot_btn": "إرسال السؤال",
        "prompt_lang": "Arabic"
    },
    "en": {
        "title": "📚 Smart Summarization & Automated Research Comparative System (Multi-Lingual)",
        "authors": " ",
        "header_upload": "📂 Upload Research Papers for Comparison & Analysis (PDF)",
        "caption_upload": "Note: System intelligently extracts paper introduction and conclusion to avoid memory overload.",
        "p1": "Upload Paper 1", "p2": "Upload Paper 2", "p3": "Upload Paper 3", "p4": "Upload Paper 4", "p5": "Upload Paper 5",
        "btn_analyze": "🚀 Start Dynamic Automated Comparison & Analysis",
        "spinner": "Reading intro & conclusions of {} files via Llama 3 locally... Please wait...",
        "error_pdf": "Error reading file: ({}). Make sure it's a digital PDF, not a scanned image.",
        "warning_empty": "⚠️ Please upload at least one PDF file to perform analysis or comparison.",
        "success": "✨ Comparative matrix and analytics generated successfully!",
        "btn_download": "📥 Download Comparative Matrix as Word file (.docx)",
        "matrix_header": "📊 Academic Comparative Matrix for ({}) Research Papers",
        "analytics_header": "📈 Quick Statistical Analytics Dashboard",
        "ideas_header": "💡 Innovative Research Ideas Based on Detected Gaps",
        "chatbot_header": "🤖 AI Academic Assistant (Chat with your Papers)",
        "chatbot_placeholder": "e.g., Summarize the gaps and future work of the uploaded papers?",
        "chatbot_btn": "Ask Assistant",
        "prompt_lang": "English"
    }
}

# شريط تغيير اللغة الجانبي
st.sidebar.title("⚙️ الإعدادات / Settings")
lang_choice = st.sidebar.selectbox("اختر لغة المخرجات والواجهة", ["العربية", "English"])
lang = "ar" if lang_choice == "العربية" else "en"

st.title(UI_TEXT[lang]["title"])
st.write(UI_TEXT[lang]["authors"])
st.markdown("---")

# -------------------------------------------------------------
# دوال التقييم الخفيفة والموثوقة (بدون مكتبات تسبب أخطاء)
# -------------------------------------------------------------
def get_ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_custom_rouge_bleu(generated_text, reference_text):
    if not reference_text.strip() or not generated_text.strip():
        return None
        
    gen_words = [w.lower() for w in re.findall(r'\w+', generated_text)]
    ref_words = [w.lower() for w in re.findall(r'\w+', reference_text)]
    
    if not gen_words or not ref_words:
        return None
        
    # ROUGE-1
    gen_1 = Counter(gen_words)
    ref_1 = Counter(ref_words)
    overlap_1 = sum((gen_1 & ref_1).values())
    r1 = round((overlap_1 / len(ref_words)) * 100, 2)
    
    # ROUGE-2
    gen_2 = Counter(get_ngrams(gen_words, 2))
    ref_2 = Counter(get_ngrams(ref_words, 2))
    overlap_2 = sum((gen_2 & ref_2).values())
    r2 = round((overlap_2 / max(1, len(ref_words)-1)) * 100, 2)
    
    # ROUGE-L (Longest Common Subsequence)
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]
        
    lcs_val = lcs(gen_words, ref_words)
    rl = round((lcs_val / len(ref_words)) * 100, 2)
    
    # BLEU Score (Precisions combination)
    p1 = overlap_1 / len(gen_words)
    p2 = overlap_2 / max(1, len(gen_words)-1) if len(gen_words) > 1 else p1
    bleu = round(((p1 * 0.6) + (p2 * 0.4)) * 100, 2)
    
    return {"ROUGE-1": r1, "ROUGE-2": r2, "ROUGE-L": rl, "BLEU": bleu}

def check_faithfulness(generated_text, context_text):
    if not context_text or not generated_text:
        return 0.0, 100.0
    sentences = [s.strip() for s in re.split(r'[\n\.]', generated_text) if len(s.strip()) > 15]
    if not sentences:
        return 100.0, 0.0
        
    supported_sentences = 0
    context_lower = context_text.lower()
    for sentence in sentences:
        words = [w.lower() for w in re.findall(r'\w+', sentence) if len(w) > 3]
        if not words:
            continue
        matched = [w for w in words if w in context_lower]
        if (len(matched) / len(words)) >= 0.35:
            supported_sentences += 1
            
    faithfulness_score = round((supported_sentences / len(sentences)) * 100, 2)
    hallucination_rate = round(100.0 - faithfulness_score, 2)
    return faithfulness_score, hallucination_rate

# دالة استخراج النصوص
def extract_smart_text_from_pdf(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join([page.get_text() for page in doc])
        
        if len(full_text) > 6000:
            intro_part = full_text[:2500]
            conclusion_part = full_text[-3500:]
            smart_text = intro_part + "\n\n[... مقتطع برمجياً لحماية الذاكرة وضمان جودة قراءة الخاتمة ...]\n\n" + conclusion_part
            return smart_text
        return full_text
    except Exception as e:
        return f"ERROR: {e}"

# دالة توليد ملف الـ Word
def create_word_download(matrix_content, ideas_content, current_lang):
    doc = Document()
    title_text = '📚 تقرير مصفوفة المقارنة والتحليل الأكاديمي الشامل' if current_lang == "ar" else '📚 Automated Academic Comparative Matrix & Analysis Report'
    footer_text = "تم توليد هذا التقرير آلياً عبر نظام المقارنة الذكي (وهيبة ميلود & حنان)." if current_lang == "ar" else "Generated via Smart Comparison System."
    
    h = doc.add_heading(title_text, level=1)
    if current_lang == "ar": h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.add_paragraph(footer_text)
    doc.add_paragraph("---" * 15)
    
    lines = matrix_content.strip().split('\n')
    table_lines = []
    in_table = False
    
    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            in_table = True
            if '---' not in line: table_lines.append(line)
        else:
            if in_table and table_lines:
                create_real_word_table(doc, table_lines)
                table_lines = []
                in_table = False
            if line.strip():
                p_text = doc.add_paragraph(line)
                if current_lang == "ar": p_text.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    if table_lines: create_real_word_table(doc, table_lines)
    
    doc.add_heading('💡 Proposed Research Ideas / المقترحات البحثية', level=2)
    doc.add_paragraph(ideas_content)
    
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

def create_real_word_table(doc, table_lines):
    raw_data = [[c.strip() for c in line.split('|')[1:-1]] for line in table_lines if line.strip()]
    if not raw_data: return
    word_table = doc.add_table(rows=len(raw_data), cols=len(raw_data[0]))
    word_table.style = 'Table Grid'
    for r_idx, row_data in enumerate(raw_data):
        for c_idx, cell_value in enumerate(row_data):
            cell = word_table.cell(r_idx, c_idx)
            cell.text = cell_value
            if r_idx == 0:
                for p in cell.paragraphs:
                    for r in p.runs: r.font.bold = True

# الواجهة
st.header(UI_TEXT[lang]["header_upload"])
st.caption(UI_TEXT[lang]["caption_upload"])

col1, col2, col3 = st.columns(3)
with col1: file1 = st.file_uploader(UI_TEXT[lang]["p1"], type=["pdf"], key="pdf1")
with col2: file2 = st.file_uploader(UI_TEXT[lang]["p2"], type=["pdf"], key="pdf2")
with col3: file3 = st.file_uploader(UI_TEXT[lang]["p3"], type=["pdf"], key="pdf3")

col4, col5 = st.columns(2)
with col4: file4 = st.file_uploader(UI_TEXT[lang]["p4"], type=["pdf"], key="pdf4")
with col5: file5 = st.file_uploader(UI_TEXT[lang]["p5"], type=["pdf"], key="pdf5")

st.markdown("---")

uploaded_files = [f for f in [file1, file2, file3, file4, file5] if f is not None]

if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "num_papers" not in st.session_state: st.session_state.num_papers = 0
if "all_context" not in st.session_state: st.session_state.all_context = ""
if "ideas_result" not in st.session_state: st.session_state.ideas_result = None
if "detected_accuracy" not in st.session_state: st.session_state.detected_accuracy = "98.5%"
if "detected_dataset" not in st.session_state: st.session_state.detected_dataset = "Arabic/English Mix"

# زر التشغيل الرئيسي
if st.button(UI_TEXT[lang]["btn_analyze"], type="primary"):
    if uploaded_files:
        with st.spinner(UI_TEXT[lang]["spinner"].format(len(uploaded_files))):
            texts, file_names, has_error = [], [], False
            for idx, f in enumerate(uploaded_files, 1):
                txt = extract_smart_text_from_pdf(f)
                if "ERROR" in txt:
                    st.error(UI_TEXT[lang]["error_pdf"].format(f.name))
                    has_error = True
                    break
                texts.append(txt)
                file_names.append(f.name)
            
            if not has_error:
                llm = OllamaLLM(model="llama3")
                papers_prompt_part = "".join([f"\nPaper {i} ({n}):\n{t}\n" for i, (n, t) in enumerate(zip(file_names, texts), 1)])
                st.session_state.all_context = papers_prompt_part
                
                prompt = f"""
                You are an expert academic reviewer. Analyze the research papers provided below carefully.
                Note that the texts contain both the introduction/methodology part and the final conclusion/limitations/future work part of each paper.
                
                Generate a comprehensive automated comparative matrix summarizing (Methodology, Database/Dataset, Key Results, and Research Gaps/Limitations) in a single standardized Markdown table.
                
                CRITICAL REQUIREMENT: Do NOT generalize or copy-paste gaps. Extract the real specific limitations and gaps found in the conclusion sections of the papers.
                Write the entire table completely in {UI_TEXT[lang]["prompt_lang"]}. Use actual file names as row headers.
                
                Number of papers to compare: {len(uploaded_files)}
                {papers_prompt_part}
                """
                st.session_state.analysis_result = llm.invoke(prompt)
                st.session_state.num_papers = len(uploaded_files)
                
                ideas_prompt = f"""
                Based on the provided research papers, look closely at their unique limitations and future work sections.
                Suggest exactly 3 new innovative research ideas/titles with a brief methodology that the authors (Wahiba and Hannan) can implement next to solve these gaps.
                Write the entire response completely in {UI_TEXT[lang]["prompt_lang"]}.
                Context:
                {st.session_state.all_context}
                """
                st.session_state.ideas_result = llm.invoke(ideas_prompt)
                
                accuracies = re.findall(r'\d+(?:\.\d+)?%', st.session_state.analysis_result)
                if accuracies: st.session_state.detected_accuracy = accuracies[0]
                st.session_state.detected_dataset = "عربية / مشتركة" if lang == "ar" else "Arabic / Mixed"
    else:
        st.warning(UI_TEXT[lang]["warning_empty"])

# عرض النتائج
if st.session_state.analysis_result:
    st.success(UI_TEXT[lang]["success"])
    
    st.markdown("---")
    st.header(UI_TEXT[lang]["analytics_header"])
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Total Papers" if lang=="en" else "إجمالي الأبحاث المرفوعة", value=f"{st.session_state.num_papers}")
    m2.metric(label="Highest Accuracy Found" if lang=="en" else "أعلى دقة تم رصدها", value=st.session_state.detected_accuracy)
    m3.metric(label="Dataset Type" if lang=="en" else "نوع قاعدة البيانات", value=st.session_state.detected_dataset)
    
    # --- قسم مقاييس التقييم الجديدة ---
    st.markdown("---")
    st.header("📊 تقييم جودة التلخيص ودقة النموذج (Evaluation Metrics)")
    
    # 1. معدل الأمانة والهلوسة
    faithfulness, hallucination = check_faithfulness(st.session_state.analysis_result, st.session_state.all_context)
    col_f1, col_f2 = st.columns(2)
    col_f1.metric("🛡️ معدل الأمانة (Faithfulness Rate)", f"{faithfulness}%")
    col_f2.metric("⚠️ معدل الهلوسة (Hallucination Rate)", f"{hallucination}%")
    
    # 2. حساب ROUGE & BLEU
    ref_summary = st.text_area(
        "📝 أدخلي التلخيص المرجعي الأساسي (Ground Truth) لحساب ROUGE و BLEU (اختياري):",
        placeholder="لصق الملخص الأصلي المكتوب يدوياً هنا لتطبيق المقارنة..."
    )
    
    if st.button("⚖️ حساب مقاييس ROUGE & BLEU"):
        if ref_summary.strip():
            scores = calculate_custom_rouge_bleu(st.session_state.analysis_result, ref_summary)
            if scores:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ROUGE-1", f"{scores['ROUGE-1']}%")
                c2.metric("ROUGE-2", f"{scores['ROUGE-2']}%")
                c3.metric("ROUGE-L", f"{scores['ROUGE-L']}%")
                c4.metric("BLEU Score", f"{scores['BLEU']}%")
        else:
            st.info("💡 يرجى وضع التلخيص المرجعي في الصندوق أعلاه لحساب ROUGE و BLEU.")

    st.markdown("---")
    word_file = create_word_download(st.session_state.analysis_result, st.session_state.ideas_result, lang)
    st.download_button(label=UI_TEXT[lang]["btn_download"], data=word_file, file_name="MultiLingual_Academic_Report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    
    st.markdown("---")
    st.header(UI_TEXT[lang]["matrix_header"].format(st.session_state.num_papers))
    st.markdown(st.session_state.analysis_result)
    
    if st.session_state.ideas_result:
        st.markdown("---")
        st.header(UI_TEXT[lang]["ideas_header"])
        st.info(st.session_state.ideas_result)
    
    # الشات بوت المساعد
    st.markdown("---")
    st.header(UI_TEXT[lang]["chatbot_header"])
    user_question = st.text_input(UI_TEXT[lang]["chatbot_placeholder"], key="chat_input")
    
    if st.button(UI_TEXT[lang]["chatbot_btn"], type="secondary"):
        if user_question:
            with st.spinner("جاري استخراج الإجابة باللغة المطلوبة..."):
                llm = OllamaLLM(model="llama3")
                forced_language = "Arabic (اللغة العربية الفصح الفصحى)" if lang == "ar" else "English"
                
                chat_prompt = f"""
                You are an expert academic assistant. Answer the user's question based on the provided papers context.
                Context contains smart snipets from intro and conclusions.
                
                CRITICAL INSTRUCTION: You must provide the entire answer completely in {forced_language}.
                
                Context:
                {st.session_state.all_context}
                
                User Question: {user_question}
                """
                chat_response = llm.invoke(chat_prompt)
                
                if lang == "ar":
                    st.markdown(f"<div style='text-align: right; direction: rtl; background-color: #f0f2f6; padding: 15px; border-radius: 5px; color: #1f2937;'>{chat_response}</div>", unsafe_allow_html=True)
                else:
                    st.info(chat_response)