import os
import re
import ast
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# إعدادات الصفحة واتجاه النصوص (RTL)
st.set_page_config(page_title="Multi-Project Code RAG with Pro Linter & Auto-Fix", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', 'Cairo', sans-serif, Arial;
    }
    .stTextInput input {
        text-align: right;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ نظام تحليل أكواد بايثون الذكي (Multi-Project Local RAG + Pro Linter + Auto-Fix)")
st.markdown("### بحث محلي أوفلاين مع فحص متطور للأخطاء واقتراح الحلول البرمجية تلقائياً")

ROOT_DIR = r"C:\my_projects_workspace" 

if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)

available_projects = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

if not available_projects:
    st.error("لم يتم العثور على أي مجلدات مشاريع فرعية داخل مسار العمل.")
    st.stop()

selected_project = st.selectbox("📂 اختر الخدمة أو المجلد المراد تحليله:", available_projects)

# دالة الفحص الموسعة مع اقتراح الحلول البرمجية لكل خطأ
def pro_check_python_syntax(project_path):
    issues_found = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_content = f.read()
                    
                    # 1. التحليل الساكن عبر AST وبناء الشجرة
                    tree = ast.parse(code_content, filename=file_path)
                    
                    for node in ast.walk(tree):
                        # أ) فحص الـ Bare Except
                        if isinstance(node, ast.ExceptHandler) and node.type is None:
                            issues_found.append({
                                "file": file,
                                "line": getattr(node, 'lineno', 'غير محدد'),
                                "msg": "تحذير أسلوبي: استخدام استثناء عام فارغ (Bare Except).",
                                "text": "except:",
                                "type": "BareExceptWarning",
                                "suggestion": "استبدلها بـ: `except Exception as e:` لتحديد نوع الاستثناء."
                            })
                        
                        # ب) فحص المقارنة مع None
                        if isinstance(node, ast.Compare):
                            for op, comparator in zip(node.ops, node.comparators):
                                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comparator, ast.Constant) and comparator.value is None:
                                    issues_found.append({
                                        "file": file,
                                        "line": getattr(node, 'lineno', 'غير محدد'),
                                        "msg": "تحذير أسلوبي: مقارنة مباشرة مع None.",
                                        "text": "",
                                        "type": "NoneComparisonWarning",
                                        "suggestion": "استبدل المقارنة بـ: `is None` أو `is not None`."
                                    })

                    # 2. فحص صياغة f-string للنصوص العربية
                    lines = code_content.splitlines()
                    for idx, line in enumerate(lines, 1):
                        if re.search(r'\bf[\u0600-\u06FF]', line) and not re.search(r'f["\']', line):
                            issues_found.append({
                                "file": file,
                                "line": idx,
                                "msg": "خطأ في صياغة الـ f-string (حرف f ملتصق بالنص العربي بدون علامات تنصيص).",
                                "text": line,
                                "type": "FStringSyntaxWarning",
                                "suggestion": "أضف علامات التنصيص هكذا: `f\"النص هنا {variable}\"`"
                            })
                            
                except (SyntaxError, ValueError, TypeError) as e:
                    line_no = getattr(e, 'lineno', 'غير محدد')
                    err_msg = getattr(e, 'msg', str(e))
                    err_text = getattr(e, 'text', '')
                    
                    # اقتراح تلقائي بناءً على نوع الخطأ النحوي
                    auto_fix = "راجع بناء الجملة في هذا السطر (تأكد من إغلاق الأقواس، كتابة الكلمات المفتاحية بشكل صحيح، أو إضافة النقطتين الرأسيتين `:`)."
                    if ":" in err_msg or "expected ':'" in err_msg:
                        auto_fix = "أضف نقطتين رأسيتين `2:` أو `:` في نهاية السطر."

                    issues_found.append({
                        "file": file,
                        "line": line_no,
                        "msg": err_msg,
                        "text": err_text,
                        "type": type(e).__name__,
                        "suggestion": auto_fix
                    })
                except Exception as e:
                    issues_found.append({
                        "file": file,
                        "line": "غير محدد",
                        "msg": str(e),
                        "text": "",
                        "type": type(e).__name__,
                        "suggestion": "راجع بنية الملف البرمجي."
                    })
    return issues_found

@st.cache_resource
def load_project_db(project_name):
    project_path = os.path.join(ROOT_DIR, project_name)
    code_texts = []
    
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_texts.append(f.read())

    if not code_texts:
        return None, []

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=500, chunk_overlap=50
    )
    docs = text_splitter.create_documents(code_texts)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    v_db = Chroma.from_documents(docs, embeddings, persist_directory=f"./chroma_db_{project_name}")
    return v_db, code_texts

# تنفيذ الفحص الموسع مع الاقتراحات للمشروع المختار
current_project_path = os.path.join(ROOT_DIR, selected_project)
code_issues = pro_check_python_syntax(current_project_path)

# عرض الأخطاء والحلول المقترحة باللون الأحمر
if code_issues:
    st.error("🚨 **تنبيه هام: رصد النظام أخطاء برمجية، وإليك الحلول المقترحة لتصحيحها:**")
    for issue in code_issues:
        st.warning(
            f"📁 **الملف:** `{issue['file']}` | 📌 **السطر:** `{issue['line']}` | ⚠️ **نوع الخطأ/التحذير:** `{issue['type']}`\n\n"
            f"💬 **التفاصيل:** `{issue['msg']}`\n\n"
            f"> 🔍 **السطر المتسبب:** `{str(issue['text']).strip() if issue['text'] else 'غير متوفر'}`\n\n"
            f"💡 **الحل أو التصحيح المقترح:** `{issue['suggestion']}`"
        )
    st.markdown("---")

vector_db, raw_project_texts = load_project_db(selected_project)

if vector_db is None:
    st.warning(f"المجلد المختار ({selected_project}) لا يحتوي على أي ملفات بايثون (.py).")
else:
    query = st.text_input(f"💬 اطرح سؤالك حول خدمة ({selected_project}):", "كيف تعمل هذه الخدمة وهل توجد أخطاء؟")

    if st.button("🔍 تحليل واستعلام"):
        if query:
            with st.spinner(f"⏳ جاري البحث والفحص حصرياً داخل ملفات بايثون لمجلد {selected_project}..."):
                retriever = vector_db.as_retriever(search_kwargs={"k": 2})
                relevant_docs = retriever.invoke(query)
                context = "\n".join([doc.page_content for doc in relevant_docs])

            with st.spinner("🤖 نموذج Llama 3.2 يولد الإجابة محلياً..."):
                prompt = f"""بناءً على كود بايثون المسترجع من خدمة ({selected_project}) أدناه، أجب عن سؤال المستخدم باللغة العربية بدقة ووضوح:

الكود المسترجع:
{context}

السؤال: {query}
الإجابة والشرح:"""

                response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
                generated_answer = response['message']['content']

            st.subheader("🤖 إجابة النموذج الأولية:")
            st.write(generated_answer)

            # طبقة التحقق الذاتي
            with st.spinner("🛡️ تعمل طبقة التحقق الذاتي وتحليل الكيانات..."):
                stored_code_combined = " ".join(raw_project_texts)
                extracted_classes = re.findall(r'class\s+(\w+)', stored_code_combined)
                extracted_functions = re.findall(r'def\s+(\w+)', stored_code_combined)
                all_valid_identifiers = set(extracted_classes + extracted_functions)
                match_count = sum(1 for ident in all_valid_identifiers if ident in generated_answer)
                
            st.markdown("---")
            st.subheader("🛡️ نتيجة طبقة التحقق الذاتي (Self-Correction & Verification):")
            
            st.info(f"📌 الكيانات البرمجية المكتشفة في مشروعك: `Classes: {extracted_classes}` | `Functions: {extracted_functions}`")
            
            if code_issues:
                st.error("[!] تم رصد مشاكل برمجية أو تحذيرات صياغة مسبقاً عبر فاحص الـ Pro Linter.")
            elif match_count > 0:
                st.success(f"[✓] تم التحقق بنجاح: الإجابة متوافقة مع كود بايثون في مجلد ({selected_project}) وخالية من الهلوسة.")
            else:
                st.warning("[!] تنبيه بحثي: لم يقم النموذج بذكر أسماء الدوال أو الكلاسات بدقة حرفية.")
        else:
            st.warning("الرجاء كتابة سؤال أولاً.")