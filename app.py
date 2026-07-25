import html
import io
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.intelligence import DocumentIntelligenceAnalyzer
from src.nlp.analyzer import NLPAnalyzer
from src.rag.pipeline import RAGPipeline
from src.report.generator import ReportGenerator
from src.utils.file_handler import FileHandler


st.set_page_config(
    page_title="EDUBREIF AI",
    page_icon="B",
    layout="wide",
)


css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path, encoding="utf-8", errors="ignore") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_state():
    defaults = {
        "rag_pipeline": None,
        "chat_history": [],
        "documents_loaded": False,
        "nlp_results": None,
        "document_intelligence": None,
        "processed_documents": [],
        "uploaded_filenames": [],
        "last_report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def sidebar():
    with st.sidebar:
        if Path("assets/logo.png").exists():
            st.image("assets/logo.png", width=60)
        else:
            st.markdown("### EDUBREIF AI")

        st.title("EDUBREIF AI")
        st.caption("Upload -> Intelligence -> Study -> Ask -> Export")
        st.divider()

        st.subheader("Settings")
        groq_key = st.text_input("Groq API Key", type="password")
        model = st.selectbox(
            "Model",
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        )

        st.divider()
        st.subheader("Upload")
        files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "csv", "md", "json"],
            accept_multiple_files=True,
        )
        process_clicked = st.button("Process", use_container_width=True)
        return groq_key, model, files, process_clicked


def process_documents(files, key, model):
    handler = FileHandler()
    documents = handler.process_files(files)

    pipeline = RAGPipeline(groq_key=key, model=model)
    pipeline.build_index(documents)

    analyzer = NLPAnalyzer()
    full_text = " ".join([doc["content"] for doc in documents])
    nlp = analyzer.analyze(full_text)

    intelligence = DocumentIntelligenceAnalyzer().analyze(documents, nlp)

    st.session_state.rag_pipeline = pipeline
    st.session_state.nlp_results = nlp
    st.session_state.document_intelligence = intelligence
    st.session_state.processed_documents = documents
    st.session_state.documents_loaded = True
    st.session_state.chat_history = []
    st.session_state.last_report = None
    st.session_state.uploaded_filenames = [f.name for f in files]


def dashboard_ui():
    intel = st.session_state.document_intelligence
    dashboard = intel["dashboard"]
    health = intel["health"]

    st.subheader("Document Dashboard")
    cols = st.columns(5)
    cols[0].metric("Title", dashboard["title"][:38])
    cols[1].metric("Type", dashboard["document_type"])
    cols[2].metric("Pages", dashboard["pages"])
    cols[3].metric("Words", f"{dashboard['word_count']:,}")
    cols[4].metric("Reading Time", dashboard["reading_time"])

    cols = st.columns(5)
    cols[0].metric("Language", dashboard["language"])
    cols[1].metric("Difficulty", dashboard["reading_difficulty"])
    cols[2].metric("Tables", dashboard["tables"])
    cols[3].metric("Figures", dashboard["figures"])
    cols[4].metric("References", dashboard["references"])
    st.metric("Sections", dashboard["sections"])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("AI Document Classification")
        scores = {k: v for k, v in dashboard["classification_scores"].items() if v > 0}
        st.bar_chart(scores or {"Unknown": 1})

    with right:
        st.subheader("Document Health Score")
        for label, score in health.items():
            st.write(f"{label}: {score}%")
            st.progress(score / 100)

    st.subheader("Knowledge Graph")
    st.code(format_tree(intel["knowledge_graph"]), language="text")


def learning_ui():
    learning = st.session_state.document_intelligence["learning"]
    st.subheader("Learning Assistant")

    notes, cards, questions, coach = st.tabs(["Notes", "Flashcards", "Questions", "Study Coach"])
    with notes:
        render_list("Short Notes", learning["short_notes"])
        render_list("Revision Notes", learning["revision_notes"])
        render_list("Last Minute Notes", learning["last_minute_notes"])
        render_list("Formula Sheet", learning["formula_sheet"] or ["No formulas detected."])
        render_list("Cheat Sheet", learning["cheat_sheet"])
        render_list("Memory Tricks", learning["memory_tricks"])
        render_list("Mnemonics", learning["mnemonics"] or ["No mnemonic generated."])
        st.subheader("Mind Map")
        st.code(format_tree(learning["mind_map"]), language="text")
        st.subheader("Concept Tree")
        st.code(format_tree(learning["concept_tree"]), language="text")

    with cards:
        for index, card in enumerate(learning["flashcards"], start=1):
            with st.expander(f"Flashcard {index}: {card['front']}"):
                st.write(card["back"])

    with questions:
        for taxonomy, qs in learning["bloom_questions"].items():
            render_list(taxonomy, qs)
        st.divider()
        for level, qs in learning["difficulty_questions"].items():
            render_list(level, qs)
        st.info(learning["adaptive_quiz"]["rule"])
        render_list("Adaptive Quiz Start", list(learning["adaptive_quiz"].values())[1:])

    with coach:
        study = learning["study_coach"]
        render_list("Summary", study["summary"])
        render_list("Smart Notes", study["smart_notes"])
        render_list("Important Questions", study["important_questions"])
        render_list("Quiz", study["quiz"])
        render_list("Revision Plan", study["revision_plan"])
        st.metric("Flashcards Created", study["flashcard_count"])


def research_ui():
    research = st.session_state.document_intelligence["research"]
    st.subheader("Research Assistant")

    fields = [
        ("Abstract", "abstract"),
        ("Research Gap", "research_gap"),
        ("Novelty", "novelty"),
        ("Dataset", "dataset"),
        ("Models Used", "models_used"),
        ("Hyperparameters", "hyperparameters"),
        ("Results", "results"),
        ("Future Work", "future_work"),
    ]
    for label, key in fields:
        with st.expander(label, expanded=label == "Abstract"):
            st.write(research[key])

    st.subheader("Paper Comparison")
    st.dataframe(research["paper_comparison"], use_container_width=True)

    st.subheader("Research Timeline")
    timeline = research["research_timeline"]
    st.code("\n↓\n".join(timeline) if timeline else "No years detected.", language="text")

    st.subheader("Citation Checker")
    st.json(research["citation_checker"])


def analytics_ui():
    analytics = st.session_state.document_intelligence["analytics"]
    st.subheader("Analytics")

    left, right = st.columns(2)
    with left:
        st.write("Top Keywords")
        st.bar_chart(dict(analytics["top_keywords"]))
        st.write("Topic Distribution")
        st.bar_chart(analytics["topic_distribution"] or {"No topics": 0})
        st.write("Entities")
        st.bar_chart(analytics["entities"] or {"No entities": 0})
    with right:
        st.write("Sentiment")
        st.bar_chart(analytics["sentiment"] or {"neutral": 1})
        st.write("Section Length")
        st.bar_chart(analytics["section_length"])
        st.write("Reading Difficulty")
        st.info(analytics["reading_difficulty"])

    st.subheader("Word Cloud Terms")
    st.write(", ".join(analytics["word_cloud"]) or "No keywords available.")

    st.subheader("Keyword Network")
    st.dataframe(analytics["keyword_network"], use_container_width=True)

    st.subheader("Document Similarity")
    if analytics["document_similarity"]:
        st.dataframe(analytics["document_similarity"], use_container_width=True)
    else:
        st.info("Upload multiple documents to compute similarity.")

    st.subheader("Timeline")
    st.code("\n".join(analytics["timeline"]) if analytics["timeline"] else "No timeline dates detected.", language="text")


def smart_ai_ui():
    smart = st.session_state.document_intelligence["smart_ai"]
    st.subheader("Smart AI")

    confidence = smart["confidence_meter"]
    st.metric("AI Confidence Meter", f"{confidence['default_confidence']}%")
    st.caption(confidence["reason"])

    st.subheader("Hallucination Detector")
    render_list("Before Answering", smart["hallucination_detector"])

    st.subheader("Explain Button Modes")
    render_list("Available Modes", smart["explain_modes"])

    st.subheader("AI Critic")
    for issue, findings in smart["ai_critic"].items():
        render_list(issue, findings or ["No obvious issue detected."])

    st.subheader("Ask Follow-up")
    render_list("You may also ask", smart["follow_up_questions"])


def chat_ui():
    st.subheader("Ask Questions")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask from the uploaded document...")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        result = st.session_state.rag_pipeline.query(
            prompt,
            chat_history=st.session_state.chat_history[:-1],
        )
        st.markdown(result["answer"])
        with st.expander("Explain Retrieval"):
            for source in result["sources"]:
                st.write(
                    f"{source['file']} | page {source.get('page_number') or 'unavailable'} "
                    f"| chunk {source['chunk_id']} | score {source.get('retrieval_score')}"
                )
                st.caption(source.get("retrieval_reason") or "No reason available.")
                st.code(source["snippet"], language="text")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
        })


def export_ui():
    st.subheader("Export")
    name = st.text_input("Document Set Name", value="EDUBREIF AI Document Package")
    author = st.text_input("Author")

    if st.button("Generate Export Package"):
        gen = ReportGenerator(
            rag_pipeline=st.session_state.rag_pipeline,
            nlp_results=st.session_state.nlp_results,
        )
        report = gen.generate(
            project_name=name,
            author=author,
            date=str(datetime.today().date()),
            filenames=st.session_state.uploaded_filenames,
        )
        st.session_state.last_report = report["markdown"]

    markdown_text = st.session_state.last_report or build_study_guide_markdown(name)
    st.download_button("Download Markdown", markdown_text, "edubrief-report.md", "text/markdown")
    st.download_button("Download HTML", markdown_to_html(markdown_text), "edubrief-report.html", "text/html")
    st.download_button(
        "Download DOCX",
        build_docx(markdown_text),
        "edubrief-report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.download_button(
        "Download PPT",
        build_pptx(markdown_text),
        "edubrief-slides.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    try:
        gen = ReportGenerator(st.session_state.rag_pipeline, st.session_state.nlp_results)
        st.download_button("Download PDF", gen.to_pdf(markdown_text), "edubrief-report.pdf", "application/pdf")
    except Exception as exc:
        st.warning(f"PDF export unavailable: {exc}")


def nlp_ui():
    res = st.session_state.nlp_results
    st.subheader("Classic NLP")
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", res["word_count"])
    col2.metric("Unique Terms", res["unique_terms"])
    col3.metric("Sentences", res["sentence_count"])
    render_list("Keywords", [f"{k} ({s:.2f})" for k, s in res["keywords"][:15]])
    render_list("Key Phrases", res["key_phrases"])
    st.info(res["summary"])


def render_list(title, items):
    st.write(f"**{title}**")
    for item in items:
        st.write(f"- {item}")


def format_tree(graph):
    lines = []
    for root, children in graph.items():
        lines.append(str(root))
        for child in children:
            lines.append(f"   |-- {child}")
    return "\n".join(lines) if lines else "No graph generated."


def build_study_guide_markdown(name):
    intel = st.session_state.document_intelligence
    learning = intel["learning"]
    lines = [
        f"# {name}",
        "",
        "## Summary",
        *[f"- {item}" for item in learning["short_notes"]],
        "",
        "## Flashcards",
    ]
    for card in learning["flashcards"]:
        lines.extend([f"- Q: {card['front']}", f"  A: {card['back']}"])
    lines.extend(["", "## Revision Plan"])
    lines.extend([f"- {item}" for item in learning["study_coach"]["revision_plan"]])
    return "\n".join(lines)


def markdown_to_html(markdown_text):
    body = "<br>".join(html.escape(line) for line in markdown_text.splitlines())
    return f"<!doctype html><html><head><meta charset='utf-8'><title>EDUBREIF AI</title></head><body>{body}</body></html>"


def build_docx(text):
    paragraphs = "".join(f"<w:p><w:r><w:t>{html.escape(line)}</w:t></w:r></w:p>" for line in text.splitlines())
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", DOCX_CONTENT_TYPES)
        docx.writestr("_rels/.rels", DOCX_RELS)
        docx.writestr("word/document.xml", document)
    return buffer.getvalue()


def build_pptx(text):
    title = next((line.strip("# ") for line in text.splitlines() if line.startswith("#")), "EDUBREIF AI")
    bullets = [line.strip("- ") for line in text.splitlines() if line.startswith("- ")][:5]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as pptx:
        pptx.writestr("[Content_Types].xml", PPTX_CONTENT_TYPES)
        pptx.writestr("_rels/.rels", PPTX_ROOT_RELS)
        pptx.writestr("ppt/presentation.xml", PPTX_PRESENTATION)
        pptx.writestr("ppt/_rels/presentation.xml.rels", PPTX_PRESENTATION_RELS)
        pptx.writestr("ppt/slides/slide1.xml", ppt_slide_xml(title, bullets))
        pptx.writestr("ppt/slides/_rels/slide1.xml.rels", EMPTY_RELS)
    return buffer.getvalue()


def ppt_slide_xml(title, bullets):
    bullet_xml = "".join(
        f"<a:p><a:r><a:t>{html.escape(bullet)}</a:t></a:r></a:p>"
        for bullet in bullets
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>
<p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{html.escape(title)}</a:t></a:r></a:p></p:txBody></p:sp>
<p:sp><p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{bullet_xml}</p:txBody></p:sp>
</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

DOCX_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

PPTX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"""

PPTX_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""

PPTX_PRESENTATION = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="9144000" cy="5143500"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>"""

PPTX_PRESENTATION_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""

EMPTY_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""


def main():
    key, model, files, process = sidebar()

    if process:
        if not key:
            st.warning("Enter API key")
        elif not files:
            st.warning("Upload files")
        else:
            try:
                with st.spinner("Processing documents and building study package..."):
                    process_documents(files, key, model)
                st.success(f"{len(files)} files processed.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if not st.session_state.documents_loaded:
        st.title("EDUBREIF AI Document Intelligence Assistant")
        st.info("Upload documents to automatically generate a dashboard, study coach, analytics, research extraction, and grounded Q&A.")
        return

    tabs = st.tabs([
        "Dashboard",
        "Learning",
        "Research",
        "Analytics",
        "Smart AI",
        "Chat",
        "Classic NLP",
        "Export",
    ])

    with tabs[0]:
        dashboard_ui()
    with tabs[1]:
        learning_ui()
    with tabs[2]:
        research_ui()
    with tabs[3]:
        analytics_ui()
    with tabs[4]:
        smart_ai_ui()
    with tabs[5]:
        chat_ui()
    with tabs[6]:
        nlp_ui()
    with tabs[7]:
        export_ui()


if __name__ == "__main__":
    main()
