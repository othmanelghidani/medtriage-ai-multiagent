import os
import sys
import json
import tempfile
import textwrap
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator_agent import OrchestratorAgent
from tools.human_validation_tool import human_validation_tool
from tools.pdf_report_generator import generate_pdf_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(BASE_DIR, "logs", "system_logs.json")

st.set_page_config(page_title="MedTriage AI", page_icon="MT", layout="wide", initial_sidebar_state="expanded")


def H(html):
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def load_logs():
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def safe_float(value):
    try:
        value = float(str(value).replace("%", "").strip())
        return value if value <= 1 else value / 100
    except Exception:
        return None


def format_confidence(value):
    cf = safe_float(value)
    if cf is None:
        return "N/A", 0
    return f"{cf * 100:.1f}%", min(cf * 100, 100)


def pred_class(pred):
    return "danger" if "pneumonia" in str(pred).lower() else "success"


def short_risk_label(risk):
    r = str(risk).lower()
    if "high" in r:
        return "High Risk"
    if "medium" in r:
        return "Medium Risk"
    if "uncertain" in r:
        return "Uncertain Risk"
    if "low" in r:
        return "Low Risk"
    return "Unknown"


def risk_class(risk):
    r = str(risk).lower()
    if "high" in r:
        return "danger"
    if "medium" in r or "uncertain" in r:
        return "warning"
    return "success"


def get_metrics(logs):
    completed = [
        x for x in logs
        if x.get("agent") == "Orchestrator Agent"
        and x.get("action") == "Workflow completed successfully"
    ]
    errors = [x for x in logs if x.get("error")]

    normal = pneumonia = 0
    confs = []

    for log in completed:
        result = log.get("result") or {}
        pred = str(result.get("prediction", "")).upper()
        conf = safe_float(result.get("confidence"))

        if "NORMAL" in pred:
            normal += 1
        elif "PNEUMONIA" in pred:
            pneumonia += 1

        if conf is not None:
            confs.append(conf)

    return {
        "total": len(completed),
        "normal": normal,
        "pneumonia": pneumonia,
        "errors": len(errors),
        "avg": sum(confs) / len(confs) if confs else 0,
    }


def render_report(report):
    labels = {
        "analysis_type": "Analysis Type",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "risk_level": "Risk Level",
        "recommendation": "Recommendation",
        "medical_disclaimer": "Medical Disclaimer",
        "human_validation_status": "Validation Status",
        "reviewer_name": "Reviewer",
        "review_timestamp": "Review Date",
        "final_status": "Final Status",
        "generation_mode": "Generation Mode",
        "timestamp": "Timestamp",
    }

    for key, value in report.items():
        label = labels.get(key, key.replace("_", " ").title())

        if key == "risk_level":
            value = short_risk_label(value)

        if isinstance(value, list):
            rows = "".join([f"<li>{item}</li>" for item in value])
            H(f"""
            <div class="report-item">
              <div class="label">{label}</div>
              <ul>{rows}</ul>
            </div>
            """)
        elif isinstance(value, dict):
            rows = "".join([f"<p><b>{k.replace('_',' ').title()}:</b> {v}</p>" for k, v in value.items()])
            H(f"""
            <div class="report-item">
              <div class="label">{label}</div>
              <div class="text">{rows}</div>
            </div>
            """)
        else:
            H(f"""
            <div class="report-item">
              <div class="label">{label}</div>
              <div class="text">{value}</div>
            </div>
            """)


def render_history(logs):
    completed = [
        x for x in logs
        if x.get("agent") == "Orchestrator Agent"
        and x.get("action") == "Workflow completed successfully"
    ][-6:][::-1]

    if not completed:
        H('<div class="muted">No completed analysis yet.</div>')
        return

    for log in completed:
        result = log.get("result") or {}
        pred = result.get("prediction", "N/A")
        risk = short_risk_label(result.get("risk_level", "N/A"))
        conf, _ = format_confidence(result.get("confidence"))

        H(f"""
        <div class="history-row">
          <div>
            <div class="mono">{log.get("timestamp", "N/A")}</div>
            <div class="muted">Workflow completed</div>
          </div>
          <span class="badge {pred_class(pred)}">{pred}</span>
          <div class="muted center">{conf}</div>
          <span class="badge {risk_class(risk)}">{risk}</span>
        </div>
        """)


def render_errors(logs):
    errors = [x for x in logs if x.get("error")][-4:][::-1]
    if not errors:
        H('<div class="muted">No recent errors.</div>')
        return

    for err in errors:
        H(f"""
        <div class="error-box">
          <b>{err.get("agent", "Agent")}</b>
          <p>{err.get("action", "")}</p>
          <small>{err.get("error", "")}</small>
        </div>
        """)


logs = load_logs()
metrics = get_metrics(logs)

H("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
    radial-gradient(circle at 20% 10%, rgba(34,211,238,.16), transparent 30%),
    radial-gradient(circle at 90% 10%, rgba(20,184,166,.16), transparent 28%),
    linear-gradient(135deg, #020617 0%, #07111f 45%, #020617 100%);
    color: #f8fafc;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 24px 34px 55px !important; max-width: 1380px !important;}

section[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, .92);
    border-right: 1px solid rgba(148,163,184,.12);
}

.card, .hero, .panel {
    background: rgba(15, 23, 42, .72);
    border: 1px solid rgba(148,163,184,.16);
    border-radius: 20px;
    box-shadow: 0 18px 55px rgba(0,0,0,.30);
    backdrop-filter: blur(18px);
}

.hero {
    padding: 22px 26px;
    margin-bottom: 16px;
    border-color: rgba(34,211,238,.25);
}

.hero-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.logo {
    width: 46px;
    height: 46px;
    border-radius: 15px;
    background: rgba(34,211,238,.16);
    border: 1px solid rgba(34,211,238,.35);
    color: #67e8f9;
    display: grid;
    place-items: center;
    font-weight: 800;
}

.title {
    font-size: 1.9rem;
    font-weight: 850;
    letter-spacing: -.04em;
}

.subtitle {
    color: #94a3b8;
    margin-top: 3px;
    font-size: .9rem;
}

.online {
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(20,184,166,.13);
    border: 1px solid rgba(45,212,191,.3);
    color: #99f6e4;
    font-size: .78rem;
    font-weight: 800;
    white-space: nowrap;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 11px;
    margin-top: 16px;
}

.metric {
    background: rgba(2,6,23,.36);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 16px;
    padding: 13px;
}

.metric .label, .label {
    color: #94a3b8;
    font-size: .66rem;
    text-transform: uppercase;
    letter-spacing: .14em;
    font-weight: 800;
    margin-bottom: 6px;
}

.metric .value {
    font-size: 1.45rem;
    font-weight: 850;
    color: #f8fafc;
}

.panel {
    padding: 16px;
    margin-bottom: 14px;
}

.panel-title {
    color: #e2e8f0;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: .14em;
    font-weight: 850;
    margin: 16px 0 10px;
}

.notice {
    background: rgba(251,191,36,.09);
    border: 1px solid rgba(251,191,36,.25);
    color: #fde68a;
    border-radius: 15px;
    padding: 11px 15px;
    font-size: .82rem;
    margin-bottom: 15px;
}

.step {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    background: rgba(2,6,23,.34);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 15px;
    padding: 11px;
    margin-bottom: 8px;
}

.step-num {
    width: 30px;
    height: 30px;
    border-radius: 10px;
    background: rgba(34,211,238,.13);
    border: 1px solid rgba(34,211,238,.3);
    color: #67e8f9;
    display: grid;
    place-items: center;
    font-weight: 850;
    flex-shrink: 0;
}

.step b {color: #f8fafc; font-size: .87rem;}
.step p {color: #94a3b8; margin: 2px 0 0; font-size: .76rem;}

.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: fit-content;
    max-width: 160px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: .68rem;
    font-weight: 850;
    text-transform: uppercase;
    letter-spacing: .05em;
    white-space: nowrap;
}

.success {
    color: #86efac;
    background: rgba(34,197,94,.12);
    border: 1px solid rgba(34,197,94,.32);
}

.danger {
    color: #fda4af;
    background: rgba(244,63,94,.12);
    border: 1px solid rgba(244,63,94,.32);
}

.warning {
    color: #fde68a;
    background: rgba(251,191,36,.12);
    border: 1px solid rgba(251,191,36,.32);
}

.conf-track {
    height: 8px;
    background: rgba(30,41,59,.85);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}

.conf-fill {
    height: 100%;
    background: linear-gradient(90deg, #22d3ee, #34d399);
    border-radius: 999px;
}

.report-item {
    background: rgba(2,6,23,.25);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 13px;
    padding: 11px;
    margin-bottom: 8px;
}

.text {
    color: #e2e8f0;
    font-size: .86rem;
    line-height: 1.5;
}

.report-item ul {
    margin: 6px 0 0 18px;
    color: #e2e8f0;
    font-size: .86rem;
}

.history-row {
    display: grid;
    grid-template-columns: 1.45fr .65fr .45fr .75fr;
    gap: 12px;
    align-items: center;
    background: rgba(2,6,23,.32);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 16px;
    padding: 12px 14px;
    margin-bottom: 10px;
}

.mono {
    color: #cbd5e1;
    font-size: .76rem;
    font-family: monospace;
}

.muted {
    color: #94a3b8;
    font-size: .76rem;
}

.center {
    text-align: center;
}

.error-box {
    background: rgba(244,63,94,.09);
    border: 1px solid rgba(244,63,94,.25);
    border-radius: 13px;
    padding: 11px;
    margin-bottom: 8px;
    color: #fda4af;
}

.error-box p, .error-box small {
    color: #cbd5e1;
    margin: 4px 0 0;
}

div.stButton > button {
    border-radius: 14px !important;
    padding: 10px 16px !important;
    min-height: 43px !important;
    font-size: .86rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0891b2, #0f766e) !important;
    color: white !important;
    border: 1px solid rgba(34,211,238,.28) !important;
}

div.stButton > button:hover {
    transform: translateY(-1px);
    border-color: rgba(103,232,249,.5) !important;
}

div.stDownloadButton > button {
    border-radius: 14px !important;
    padding: 10px 16px !important;
    min-height: 43px !important;
    background: linear-gradient(135deg, #059669, #0f766e) !important;
    color: white !important;
    font-weight: 800 !important;
    border: 1px solid rgba(52,211,153,.35) !important;
}

.stFileUploader > div {
    background: rgba(15,23,42,.60) !important;
    border: 1.5px dashed rgba(34,211,238,.28) !important;
    border-radius: 18px !important;
}

[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,.16);
}

input {
    background: rgba(2,6,23,.55) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
}

@media(max-width: 900px) {
    .metrics {grid-template-columns: 1fr 1fr;}
    .history-row {grid-template-columns: 1fr;}
}
</style>
""")

with st.sidebar:
    H(f"""
    <div class="panel">
      <div class="label">MedTriage AI</div>
      <div style="font-size:1.35rem;font-weight:850;">Online</div>
      <div class="muted">Multi-agent medical triage platform</div>
    </div>
    <div class="panel">
      <div class="label">Completed Analyses</div>
      <div style="font-size:1.55rem;font-weight:850;">{metrics["total"]}</div>
      <div class="muted">Successful orchestrated workflows</div>
    </div>
    <div class="panel">
      <div class="label">Average Confidence</div>
      <div style="font-size:1.55rem;font-weight:850;">{metrics["avg"] * 100:.1f}%</div>
      <div class="muted">CNN prediction confidence</div>
    </div>
    <div class="panel">
      <div class="label">System Errors</div>
      <div style="font-size:1.55rem;font-weight:850;">{metrics["errors"]}</div>
      <div class="muted">Validation or generation errors</div>
    </div>
    """)

H(f"""
<div class="hero">
  <div class="hero-top">
    <div class="brand">
      <div class="logo">MT</div>
      <div>
        <div class="title">MedTriage AI</div>
        <div class="subtitle">AI-assisted chest X-ray triage using multi-agent workflow, PyTorch CNN, Ollama LLM and human validation.</div>
      </div>
    </div>
    <div class="online">System Online</div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="label">Total Analyses</div>
      <div class="value">{metrics["total"]}</div>
    </div>
    <div class="metric">
      <div class="label">Pneumonia</div>
      <div class="value">{metrics["pneumonia"]}</div>
    </div>
    <div class="metric">
      <div class="label">Normal</div>
      <div class="value">{metrics["normal"]}</div>
    </div>
    <div class="metric">
      <div class="label">Avg Confidence</div>
      <div class="value">{metrics["avg"] * 100:.1f}%</div>
    </div>
  </div>
</div>

<div class="notice">
This system provides AI-assisted preliminary analysis only. It does not replace a qualified medical professional.
Human validation is required before report finalization.
</div>
""")

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    H('<div class="panel-title">Case Upload</div>')
    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    H('<div class="panel-title">Multi-Agent Timeline</div>')
    steps = [
        ("01", "Orchestrator Agent", "Coordinates the full workflow."),
        ("02", "X-ray Validator", "Rejects non-medical or invalid images."),
        ("03", "CNN Classifier", "Runs the PyTorch NORMAL/PNEUMONIA model."),
        ("04", "Report Generator", "Uses Ollama phi3 to generate a structured report."),
        ("05", "Human Validation", "Reviewer approves or rejects the report."),
        ("06", "PDF Export", "Approved report becomes downloadable."),
    ]

    for n, title, desc in steps:
        H(f"""
        <div class="step">
          <div class="step-num">{n}</div>
          <div><b>{title}</b><p>{desc}</p></div>
        </div>
        """)

with right:
    H('<div class="panel-title">Analysis Console</div>')

    if not uploaded_file:
        st.info("Upload a chest X-ray image to start the analysis.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.getbuffer())
            image_path = tmp.name

        run_col, clear_col = st.columns([2, 1], gap="small")

        with run_col:
            run_analysis = st.button("Run Multi-Agent Analysis", use_container_width=True)

        with clear_col:
            clear = st.button("Clear", use_container_width=True)

        if clear:
            for key in ["classification", "report", "final_report", "pdf_path"]:
                st.session_state.pop(key, None)
            st.rerun()

        if run_analysis:
            for key in ["classification", "report", "final_report", "pdf_path"]:
                st.session_state.pop(key, None)

            with st.spinner("Running multi-agent workflow..."):
                orchestrator = OrchestratorAgent()
                result = orchestrator.run(image_path)

            if result.get("error"):
                st.error(result["error"])
            else:
                st.session_state["classification"] = result.get("classification")
                st.session_state["report"] = result.get("report")
                st.success("Analysis completed successfully.")

    if "classification" in st.session_state:
        classification = st.session_state["classification"] or {}
        report = st.session_state["report"] or {}

        pred = classification.get("prediction", "Unknown")
        conf = classification.get("confidence", "N/A")
        conf_text, bar = format_confidence(conf)
        risk = short_risk_label(report.get("risk_level", "Pending"))

        H(f"""
        <div class="panel">
          <div class="label">Classification Result</div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span class="badge {pred_class(pred)}">{pred}</span>
            <span class="badge {risk_class(risk)}">{risk}</span>
          </div>
          <div class="text">Confidence: <b>{conf_text}</b></div>
          <div class="conf-track"><div class="conf-fill" style="width:{bar}%"></div></div>
        </div>
        """)

        H('<div class="panel-title">Generated Medical Report</div>')
        render_report(report)

if "report" in st.session_state:
    H('<div class="panel-title">Human Validation</div>')

    reviewer_name = st.text_input("Reviewer name", value="Dr Othman El Ghidani")

    approve_col, reject_col = st.columns(2, gap="medium")

    with approve_col:
        approve = st.button("Approve + Generate PDF", use_container_width=True)

    with reject_col:
        reject = st.button("Reject Report", use_container_width=True)

    if approve:
        with st.spinner("Approving report and generating PDF..."):
            vr = human_validation_tool(
                report=st.session_state["report"],
                decision="approve",
                reviewer_name=reviewer_name,
            )

        if vr.get("error"):
            st.error(vr["error"])
        else:
            st.session_state["final_report"] = vr["final_report"]
            st.session_state["pdf_path"] = generate_pdf_report(vr["final_report"])
            st.success("Report approved and PDF generated.")

    if reject:
        with st.spinner("Rejecting report..."):
            vr = human_validation_tool(
                report=st.session_state["report"],
                decision="reject",
                reviewer_name=reviewer_name,
            )

        if vr.get("error"):
            st.error(vr["error"])
        else:
            st.session_state["final_report"] = vr["final_report"]
            st.warning("Report rejected.")

if "final_report" in st.session_state:
    final_report = st.session_state["final_report"]
    status = str(final_report.get("human_validation_status", "")).lower()
    approved = status == "approved"

    H(f"""
    <div class="panel-title">
      Final Report
      <span class="badge {'success' if approved else 'danger'}" style="margin-left:10px;">
        {'Approved' if approved else 'Rejected'}
      </span>
    </div>
    """)

    render_report(final_report)

    if approved and "pdf_path" in st.session_state:
        with open(st.session_state["pdf_path"], "rb") as pdf_file:
            st.download_button(
                "Download Approved PDF",
                data=pdf_file,
                file_name=os.path.basename(st.session_state["pdf_path"]),
                mime="application/pdf",
                use_container_width=True,
            )

h1, h2 = st.columns([1.25, .75], gap="large")

with h1:
    H('<div class="panel-title">Analysis History</div>')
    render_history(load_logs())

with h2:
    H('<div class="panel-title">System Monitoring</div>')
    render_errors(load_logs())