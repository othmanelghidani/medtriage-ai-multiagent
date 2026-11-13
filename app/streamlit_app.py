import os
import sys
import json
import tempfile
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator_agent import OrchestratorAgent
from tools.human_validation_tool import human_validation_tool
from tools.pdf_report_generator import generate_pdf_report

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH  = os.path.join(BASE_DIR, "logs", "system_logs.json")

st.set_page_config(
    page_title="MedTriage AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def md(html: str):
    st.markdown(html, unsafe_allow_html=True)


def load_logs():
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def safe_float(v):
    try:
        v = float(str(v).replace("%", "").strip())
        return v if v <= 1 else v / 100
    except Exception:
        return None


def fmt_conf(v):
    cf = safe_float(v)
    if cf is None:
        return "—", 0
    return f"{cf * 100:.1f}%", min(cf * 100, 100)


def risk_info(risk):
    r = str(risk).lower()
    if "high"    in r: return "High Risk",      "#EF4444", "#FEF2F2"
    if "medium"  in r: return "Medium Risk",    "#F59E0B", "#FFFBEB"
    if "uncertain" in r: return "Uncertain",    "#8B5CF6", "#F5F3FF"
    if "low"     in r: return "Low Risk",       "#10B981", "#ECFDF5"
    return "Unknown", "#6B7280", "#F9FAFB"


def pred_info(pred):
    if "pneumonia" in str(pred).lower():
        return "#EF4444", "#FEF2F2", "🔴"
    return "#10B981", "#ECFDF5", "🟢"


def get_metrics(logs):
    done   = [x for x in logs if x.get("agent") == "Orchestrator Agent"
              and x.get("action") == "Workflow completed successfully"]
    errors = [x for x in logs if x.get("error")]
    confs  = []
    normal = pneumonia = 0
    for log in done:
        r    = log.get("result") or {}
        pred = str(r.get("prediction", "")).upper()
        cf   = safe_float(r.get("confidence"))
        if "NORMAL"    in pred: normal    += 1
        if "PNEUMONIA" in pred: pneumonia += 1
        if cf is not None:      confs.append(cf)
    return {
        "total":     len(done),
        "normal":    normal,
        "pneumonia": pneumonia,
        "errors":    len(errors),
        "avg":       sum(confs) / len(confs) if confs else 0,
    }


# ─────────────────────────────────────────────────────────────────────
# CSS  — clean clinical white with navy/teal accents
# ─────────────────────────────────────────────────────────────────────
md("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1E293B;
}

.stApp {
    background: #F8FAFC;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 28px 36px 60px !important;
    max-width: 1440px !important;
}

/* ── Sidebar ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F1F5F9 !important;
}

/* ── Buttons ───────────────────────────────────────────────── */
div.stButton > button {
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all .18s ease !important;
}
div.stButton > button[kind="primary"],
div.stButton > button:first-child {
    background: #0F172A !important;
    color: #F8FAFC !important;
    border: none !important;
}
div.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

div.stDownloadButton > button {
    border-radius: 10px !important;
    background: #10B981 !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
}

/* ── Upload zone ───────────────────────────────────────────── */
.stFileUploader > div {
    background: white !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    transition: border-color .2s !important;
}
.stFileUploader > div:hover {
    border-color: #0EA5E9 !important;
}

/* ── Text input ────────────────────────────────────────────── */
.stTextInput input {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    background: white !important;
    color: #1E293B !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.12) !important;
}

/* ── Image ─────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #E2E8F0;
}

/* ── Info / warning / error ────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
}
</style>
""")


# ─────────────────────────────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────

def section_title(text, sub=None):
    sub_html = f'<p style="margin:2px 0 0;color:#64748B;font-size:.82rem;font-weight:400;">{sub}</p>' if sub else ""
    md(f"""
    <div style="margin:24px 0 14px;">
      <h3 style="margin:0;font-size:1rem;font-weight:700;color:#0F172A;
                 text-transform:uppercase;letter-spacing:.08em;">{text}</h3>
      {sub_html}
    </div>
    """)


def stat_card(label, value, color="#0F172A", bg="#FFFFFF", border="#E2E8F0"):
    md(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:14px;
                padding:16px 18px;margin-bottom:0;">
      <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.12em;color:#64748B;margin-bottom:6px;">{label}</div>
      <div style="font-size:1.6rem;font-weight:700;color:{color};
                  font-family:'DM Sans',sans-serif;">{value}</div>
    </div>
    """)


def result_card(prediction, confidence_text, bar_pct, risk_label, risk_color, risk_bg):
    pred_color, pred_bg, pred_icon = pred_info(prediction)
    md(f"""
    <div style="background:white;border:1px solid #E2E8F0;border-radius:16px;
                padding:20px 22px;margin-bottom:14px;">

      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:18px;">
        <div>
          <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:.12em;color:#94A3B8;margin-bottom:6px;">Diagnosis</div>
          <div style="display:inline-flex;align-items:center;gap:8px;
                      background:{pred_bg};border:1px solid {pred_color}30;
                      border-radius:10px;padding:8px 14px;">
            <span style="font-size:1rem;">{pred_icon}</span>
            <span style="font-size:1rem;font-weight:700;color:{pred_color};">{prediction}</span>
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:.12em;color:#94A3B8;margin-bottom:6px;">Risk Level</div>
          <div style="display:inline-block;background:{risk_bg};
                      border:1px solid {risk_color}40;border-radius:10px;
                      padding:8px 14px;font-size:.92rem;font-weight:700;
                      color:{risk_color};">{risk_label}</div>
        </div>
      </div>

      <div style="margin-bottom:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:6px;">
          <span style="font-size:.75rem;font-weight:600;color:#64748B;">
            Model Confidence
          </span>
          <span style="font-size:.88rem;font-weight:700;color:#0F172A;">
            {confidence_text}
          </span>
        </div>
        <div style="height:6px;background:#F1F5F9;border-radius:999px;overflow:hidden;">
          <div style="height:100%;width:{bar_pct}%;
                      background:linear-gradient(90deg,#0EA5E9,#10B981);
                      border-radius:999px;transition:width .6s ease;"></div>
        </div>
      </div>
    </div>
    """)


def report_field(label, value, highlight=False):
    bg    = "#F0FDF4" if highlight else "#F8FAFC"
    border = "#86EFAC" if highlight else "#E2E8F0"
    md(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
                padding:12px 16px;margin-bottom:8px;">
      <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.1em;color:#94A3B8;margin-bottom:4px;">{label}</div>
      <div style="font-size:.88rem;color:#1E293B;line-height:1.55;">{value}</div>
    </div>
    """)


def history_row(ts, pred, conf, risk):
    pred_color, pred_bg, pred_icon = pred_info(pred)
    risk_label, risk_color, risk_bg = risk_info(risk)
    md(f"""
    <div style="display:grid;grid-template-columns:1.6fr .9fr .7fr .9fr;gap:12px;
                align-items:center;background:white;border:1px solid #E2E8F0;
                border-radius:12px;padding:12px 16px;margin-bottom:8px;">
      <div>
        <div style="font-size:.8rem;font-weight:600;color:#1E293B;
                    font-family:'DM Mono',monospace;">{ts}</div>
        <div style="font-size:.72rem;color:#94A3B8;margin-top:2px;">Completed</div>
      </div>
      <div style="display:inline-flex;align-items:center;gap:5px;
                  background:{pred_bg};border-radius:8px;padding:5px 10px;
                  font-size:.75rem;font-weight:700;color:{pred_color};">
        {pred_icon} {pred}
      </div>
      <div style="font-size:.88rem;font-weight:600;color:#0F172A;text-align:center;">
        {conf}
      </div>
      <div style="display:inline-block;background:{risk_bg};border-radius:8px;
                  padding:5px 10px;font-size:.72rem;font-weight:700;color:{risk_color};">
        {risk_label}
      </div>
    </div>
    """)


def disclaimer_banner():
    md("""
    <div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:12px;
                padding:12px 16px;margin-bottom:20px;display:flex;gap:10px;
                align-items:flex-start;">
      <span style="font-size:1rem;flex-shrink:0;">⚠️</span>
      <p style="margin:0;font-size:.8rem;color:#92400E;line-height:1.5;">
        <strong>AI-Assisted Preliminary Triage Only.</strong>
        This system does not replace clinical judgment.
        All results require human validation before any medical decision.
      </p>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
logs    = load_logs()
metrics = get_metrics(logs)

with st.sidebar:
    md("""
    <div style="padding:20px 16px 10px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <span style="font-size:1.5rem;">🫁</span>
        <div>
          <div style="font-size:1rem;font-weight:700;color:#F1F5F9;">MedTriage AI</div>
          <div style="font-size:.72rem;color:#64748B;">v1.0 · Online</div>
        </div>
      </div>
    </div>
    <hr style="border:none;border-top:1px solid #1E293B;margin:0 16px 16px;">
    """)

    with st.sidebar:
        md(f"""
        <div style="padding:0 16px;">
          <div style="margin-bottom:14px;">
            <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:.12em;color:#475569;margin-bottom:4px;">
              Total Analyses
            </div>
            <div style="font-size:1.8rem;font-weight:700;color:#F1F5F9;">
              {metrics['total']}
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
            <div style="background:#1E293B;border-radius:10px;padding:12px;">
              <div style="font-size:.62rem;color:#64748B;font-weight:700;
                          text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">
                Normal
              </div>
              <div style="font-size:1.2rem;font-weight:700;color:#34D399;">
                {metrics['normal']}
              </div>
            </div>
            <div style="background:#1E293B;border-radius:10px;padding:12px;">
              <div style="font-size:.62rem;color:#64748B;font-weight:700;
                          text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">
                Pneumonia
              </div>
              <div style="font-size:1.2rem;font-weight:700;color:#F87171;">
                {metrics['pneumonia']}
              </div>
            </div>
          </div>

          <div style="background:#1E293B;border-radius:10px;padding:12px;margin-bottom:14px;">
            <div style="font-size:.62rem;color:#64748B;font-weight:700;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">
              Avg. Confidence
            </div>
            <div style="font-size:1.2rem;font-weight:700;color:#7DD3FC;">
              {metrics['avg']*100:.1f}%
            </div>
          </div>

          <div style="background:#1E293B;border-radius:10px;padding:12px;">
            <div style="font-size:.62rem;color:#64748B;font-weight:700;
                        text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;">
              System Errors
            </div>
            <div style="font-size:1.2rem;font-weight:700;
                        color:{'#F87171' if metrics['errors'] > 0 else '#34D399'};">
              {metrics['errors']}
            </div>
          </div>
        </div>
        """)

    md("<hr style='border:none;border-top:1px solid #1E293B;margin:16px;'>")
    md("""
    <div style="padding:0 16px 20px;">
      <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.12em;color:#475569;margin-bottom:10px;">
        System Info
      </div>
      <div style="font-size:.78rem;color:#64748B;line-height:1.8;">
        🤖 Framework: CrewAI<br>
        🧠 CNN: PyTorch ResNet<br>
        📝 LLM: Ollama phi3 (local)<br>
        🔒 Data: stays on device
      </div>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────
md("""
<div style="display:flex;justify-content:space-between;align-items:center;
            margin-bottom:6px;">
  <div>
    <h1 style="margin:0;font-size:1.6rem;font-weight:700;color:#0F172A;">
      Chest X-Ray Analysis
    </h1>
    <p style="margin:4px 0 0;color:#64748B;font-size:.88rem;">
      AI-assisted preliminary triage · Multi-agent workflow
    </p>
  </div>
  <div style="display:flex;align-items:center;gap:8px;
              background:#F0FDF4;border:1px solid #86EFAC;
              border-radius:999px;padding:7px 14px;">
    <div style="width:8px;height:8px;background:#22C55E;
                border-radius:50%;animation:pulse 2s infinite;"></div>
    <span style="font-size:.78rem;font-weight:700;color:#166534;">System Online</span>
  </div>
</div>
<style>
@keyframes pulse {
  0%,100% { opacity:1; }
  50%      { opacity:.4; }
}
</style>
""")

disclaimer_banner()

# ─────────────────────────────────────────────────────────────────────
# MAIN LAYOUT  — two columns
# ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.1], gap="large")


# ── LEFT: upload + image ─────────────────────────────────────────────
with left:

    section_title("X-Ray Image", "Upload a chest X-ray to begin")

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded X-Ray")

        md("<div style='margin:14px 0 6px;'></div>")

        col_run, col_clear = st.columns([3, 1], gap="small")
        with col_run:
            run_btn = st.button("▶  Run Analysis", use_container_width=True)
        with col_clear:
            clear_btn = st.button("✕ Clear", use_container_width=True)

        if clear_btn:
            for k in ["classification", "report", "final_report", "pdf_path"]:
                st.session_state.pop(k, None)
            st.rerun()

        if run_btn:
            for k in ["classification", "report", "final_report", "pdf_path"]:
                st.session_state.pop(k, None)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.getbuffer())
                image_path = tmp.name

            with st.spinner("Running multi-agent analysis…"):
                result = OrchestratorAgent().run(image_path)

            if result.get("error"):
                st.error(f"**Error:** {result['error']}")
            else:
                st.session_state["classification"] = result.get("classification")
                st.session_state["report"]          = result.get("report")
                st.success("Analysis completed.")

    else:
        md("""
        <div style="background:white;border:1px solid #E2E8F0;border-radius:14px;
                    padding:32px;text-align:center;color:#94A3B8;margin-top:8px;">
          <div style="font-size:2rem;margin-bottom:10px;">🫁</div>
          <div style="font-size:.9rem;font-weight:500;">
            Drag &amp; drop or click to upload
          </div>
          <div style="font-size:.78rem;margin-top:4px;">
            JPG, JPEG, PNG accepted
          </div>
        </div>
        """)

    # ── Recent history ──────────────────────────────────────────────
    section_title("Recent Cases", "Last 5 completed analyses")

    completed = [
        x for x in logs
        if x.get("agent") == "Orchestrator Agent"
        and x.get("action") == "Workflow completed successfully"
    ][-5:][::-1]

    if not completed:
        md("""
        <div style="background:white;border:1px solid #E2E8F0;border-radius:12px;
                    padding:18px;text-align:center;color:#94A3B8;font-size:.84rem;">
          No completed analyses yet.
        </div>
        """)
    else:
        for log in completed:
            r    = log.get("result") or {}
            conf, _ = fmt_conf(r.get("confidence"))
            history_row(
                ts   = log.get("timestamp", "—"),
                pred = r.get("prediction", "—"),
                conf = conf,
                risk = r.get("risk_level", "—"),
            )


# ── RIGHT: results + validation ──────────────────────────────────────
with right:

    if "classification" not in st.session_state:
        section_title("Analysis Results", "Results will appear here after analysis")
        md("""
        <div style="background:white;border:1px solid #E2E8F0;border-radius:14px;
                    padding:40px;text-align:center;color:#94A3B8;">
          <div style="font-size:2.5rem;margin-bottom:12px;">📋</div>
          <div style="font-size:.9rem;font-weight:500;margin-bottom:6px;">
            No results yet
          </div>
          <div style="font-size:.8rem;">Upload an X-ray and run the analysis.</div>
        </div>
        """)

    else:
        classification = st.session_state["classification"] or {}
        report         = st.session_state["report"]         or {}

        pred                       = classification.get("prediction", "Unknown")
        conf_text, bar_pct         = fmt_conf(classification.get("confidence", 0))
        risk_label, risk_color, risk_bg = risk_info(report.get("risk_level", ""))

        section_title("Analysis Results")
        result_card(pred, conf_text, bar_pct, risk_label, risk_color, risk_bg)

        # ── Report fields ───────────────────────────────────────────
        section_title("Medical Report", "Generated by Ollama phi3 (local)")

        report_field("Analysis Type",    report.get("analysis_type", "—"))
        report_field("Recommendation",   report.get("recommendation", "—"), highlight=True)
        report_field("Medical Disclaimer", report.get("medical_disclaimer", "—"))
        report_field("Generation Mode",  report.get("generation_mode", "—"))
        report_field("Timestamp",        report.get("timestamp", "—"))

        # ── Human Validation ────────────────────────────────────────
        section_title("Physician Validation", "Required before report finalization")

        md("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;
                    padding:18px 20px;margin-bottom:14px;">
        """)

        reviewer = st.text_input(
            "Reviewer name",
            value="Dr. ",
            placeholder="Dr. Full Name",
            label_visibility="visible",
        )

        col_ap, col_rj = st.columns(2, gap="small")

        with col_ap:
            approve_btn = st.button("✔  Approve & Generate PDF",
                                    use_container_width=True)
        with col_rj:
            reject_btn  = st.button("✕  Reject Report",
                                    use_container_width=True)

        md("</div>")

        if approve_btn:
            with st.spinner("Approving and generating PDF…"):
                vr = human_validation_tool(
                    report=st.session_state["report"],
                    decision="approve",
                    reviewer_name=reviewer,
                )
            if vr.get("error"):
                st.error(vr["error"])
            else:
                st.session_state["final_report"] = vr["final_report"]
                st.session_state["pdf_path"]     = generate_pdf_report(vr["final_report"])
                st.success("Report approved. PDF ready.")

        if reject_btn:
            with st.spinner("Rejecting report…"):
                vr = human_validation_tool(
                    report=st.session_state["report"],
                    decision="reject",
                    reviewer_name=reviewer,
                )
            if vr.get("error"):
                st.error(vr["error"])
            else:
                st.session_state["final_report"] = vr["final_report"]
                st.warning("Report rejected. Medical review required.")

    # ── Final report + download ──────────────────────────────────────
    if "final_report" in st.session_state:
        final   = st.session_state["final_report"]
        status  = str(final.get("human_validation_status", "")).lower()
        approved = status == "approved"

        section_title(
            "Validated Report",
            "Approved ✔" if approved else "Rejected ✗"
        )

        if approved:
            md("""
            <div style="background:#F0FDF4;border:1px solid #86EFAC;
                        border-radius:12px;padding:12px 16px;margin-bottom:12px;
                        display:flex;align-items:center;gap:10px;">
              <span style="font-size:1.1rem;">✅</span>
              <span style="font-size:.86rem;color:#166534;font-weight:600;">
                Report approved and signed. Ready for download.
              </span>
            </div>
            """)
        else:
            md("""
            <div style="background:#FEF2F2;border:1px solid #FECACA;
                        border-radius:12px;padding:12px 16px;margin-bottom:12px;
                        display:flex;align-items:center;gap:10px;">
              <span style="font-size:1.1rem;">🚫</span>
              <span style="font-size:.86rem;color:#991B1B;font-weight:600;">
                Report rejected. Requires further medical review.
              </span>
            </div>
            """)

        report_field("Reviewer",         final.get("reviewer_name", "—"))
        report_field("Review Date",      final.get("review_timestamp", "—"))
        report_field("Validation Status", final.get("human_validation_status", "—"),
                     highlight=approved)
        report_field("Final Status",     final.get("final_status", "—"))

        if approved and "pdf_path" in st.session_state:
            md("<div style='margin-top:12px;'>")
            with open(st.session_state["pdf_path"], "rb") as f:
                st.download_button(
                    "⬇  Download Medical Report PDF",
                    data=f,
                    file_name=os.path.basename(st.session_state["pdf_path"]),
                    mime="application/pdf",
                    use_container_width=True,
                )
            md("</div>")