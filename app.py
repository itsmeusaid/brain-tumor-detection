import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import logging
import os
import sqlite3
import json

from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Page Config ──
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──
st.markdown("""
<style>
/* ── Banner ── */
.ns-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    border: 1px solid rgba(255,255,255,0.05);
}
.ns-banner h1 {
    color: #ffffff !important;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    letter-spacing: 0.5px;
}
.ns-banner p {
    color: #94a3b8 !important;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Section headers ── */
.section-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #334155;
}

/* ── Result cards ── */
.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0;
    color: #e2e8f0;
}
.result-card h4 {
    color: #f1f5f9 !important;
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
}
.result-card p {
    color: #cbd5e1 !important;
    margin: 0.2rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ── Prediction badges ── */
.pred-badge {
    padding: 1rem 1.4rem;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    font-size: 1rem;
    margin: 0.6rem 0;
    color: #ffffff !important;
}
.pred-high   { background: linear-gradient(135deg, #dc2626, #b91c1c); }
.pred-medium { background: linear-gradient(135deg, #d97706, #b45309); }
.pred-low    { background: linear-gradient(135deg, #0369a1, #0284c7); }
.pred-clear  { background: linear-gradient(135deg, #047857, #059669); }

/* ── Score bars ── */
.score-row {
    display: flex;
    align-items: center;
    margin: 0.35rem 0;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.score-label { width: 110px; font-weight: 500; color: #e2e8f0; }
.score-bar-bg {
    flex: 1; height: 10px; background: #1e293b;
    border-radius: 5px; margin: 0 0.6rem;
    overflow: hidden; border: 1px solid #334155;
}
.score-bar-fill {
    height: 100%; border-radius: 5px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    transition: width 0.4s ease;
}
.score-value { width: 55px; text-align: right; font-weight: 600; color: #e2e8f0; }

/* ── History card ── */
.history-card {
    background: #1e293b;
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.6rem 0;
    border-top: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
}
.history-card .hc-title {
    font-weight: 600; font-size: 0.92rem; color: #f1f5f9; margin-bottom: 0.3rem;
}
.history-card .hc-body {
    font-size: 0.85rem; color: #94a3b8; line-height: 1.6;
}
.history-card .hc-body b { color: #e2e8f0; }

/* ── Report preview ── */
.report-preview {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    line-height: 1.65;
    color: #cbd5e1;
    white-space: pre-wrap;
    overflow-x: auto;
}

/* ── Status pill ── */
.status-saved {
    display: inline-block;
    background: #065f46;
    color: #6ee7b7;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    margin: 0.3rem 0;
}

/* ── Footer ── */
.ns-footer {
    text-align: center; color: #64748b;
    font-size: 0.78rem; padding: 1rem 0 0.5rem;
}

/* ── Button tweaks ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important; border: none !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
#  DATABASE — SQLite for multi-user persistence
# ==========================================================
DB_PATH = "neuroscan_patients.db"


def get_db():
    """Get a database connection (creates tables on first call)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age         INTEGER,
            gender      TEXT,
            mri_result  TEXT,
            mri_confidence REAL,
            mri_details TEXT,
            symptom_result TEXT,
            symptom_confidence REAL,
            symptom_text TEXT,
            symptom_details TEXT,
            report_text TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_record(patient_id, patient_name, age, gender,
                mri_data, symptom_data, report_text):
    """Save a complete diagnostic record to the database."""
    conn = get_db()
    conn.execute("""
        INSERT INTO patient_records
        (patient_id, patient_name, age, gender,
         mri_result, mri_confidence, mri_details,
         symptom_result, symptom_confidence, symptom_text, symptom_details,
         report_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        patient_name,
        age,
        gender,
        mri_data.get('prediction', '') if mri_data else None,
        mri_data.get('confidence', 0) if mri_data else None,
        json.dumps(mri_data.get('all_predictions', {})) if mri_data else None,
        symptom_data.get('prediction', '') if symptom_data else None,
        symptom_data.get('confidence', 0) if symptom_data else None,
        symptom_data.get('input_text', '') if symptom_data else None,
        json.dumps(symptom_data.get('all_scores', {})) if symptom_data else None,
        report_text,
        datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()


def search_records(patient_id="", patient_name=""):
    """Search records by patient ID or name. Returns list of dicts."""
    conn = get_db()
    query = "SELECT * FROM patient_records WHERE 1=1"
    params = []
    if patient_id.strip():
        query += " AND patient_id LIKE ?"
        params.append(f"%{patient_id.strip()}%")
    if patient_name.strip():
        query += " AND patient_name LIKE ?"
        params.append(f"%{patient_name.strip()}%")
    query += " ORDER BY created_at DESC"

    cursor = conn.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return rows


def delete_record(record_id):
    """Delete a record by its ID."""
    conn = get_db()
    conn.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def get_all_patient_ids():
    """Get unique patient IDs for dropdown."""
    conn = get_db()
    cursor = conn.execute(
        "SELECT DISTINCT patient_id, patient_name FROM patient_records ORDER BY patient_id"
    )
    results = cursor.fetchall()
    conn.close()
    return results


# ==========================================================
#  BrainTumorAnalyzer
# ==========================================================
class BrainTumorAnalyzer:

    def __init__(self):
        self.img_width, self.img_height = 224, 224
        self.class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.model_path = "my_brain_tumor_mobilenetv2.h5"

        # Train NLP in __init__
        self.symptoms_texts = [
            "headache dizziness blurred vision",
            "severe headache memory loss confusion",
            "nausea seizures vision problem",
            "seizures vomiting and nausea",
            "hormone issues weight gain fatigue",
            "growth problems infertility hormonal",
            "no headache no tumor normal",
            "healthy normal no symptoms",
        ]
        self.symptoms_labels = [
            "glioma", "glioma",
            "meningioma", "meningioma",
            "pituitary", "pituitary",
            "notumor", "notumor",
        ]
        self.tfidf = TfidfVectorizer()
        X_train = self.tfidf.fit_transform(self.symptoms_texts)
        self.text_clf = LogisticRegression(max_iter=200)
        self.text_clf.fit(X_train, self.symptoms_labels)

    @st.cache_resource
    def load_cnn_model(_self):
        try:
            if os.path.exists(_self.model_path):
                return load_model(_self.model_path)
            else:
                st.error(f"Model not found: {_self.model_path}")
                return None
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None

    def preprocess_image(self, uploaded_file):
        if uploaded_file.size > 10 * 1024 * 1024:
            raise ValueError("File too large. Max 10 MB.")
        img = Image.open(uploaded_file).resize(
            (self.img_width, self.img_height)).convert('RGB')
        return np.expand_dims(np.array(img) / 255.0, axis=0)

    def analyze_image(self, processed_image, model):
        if model is None:
            raise ValueError("Model not loaded")
        preds = model.predict(processed_image)
        idx = np.argmax(preds[0])
        return {
            'prediction': self.class_labels[idx],
            'confidence': float(preds[0][idx]) * 100,
            'all_predictions': {
                l: float(preds[0][i]) * 100
                for i, l in enumerate(self.class_labels)
            },
        }

    def predict_symptoms(self, text):
        if not text.strip():
            return {'error': 'Please enter symptoms'}
        try:
            X = self.tfidf.transform([text.lower()])
            pred = self.text_clf.predict(X)[0]
            try:
                probs = self.text_clf.predict_proba(X)[0]
                conf = max(probs) * 100
                scores = {c: p * 100 for c, p in zip(self.text_clf.classes_, probs)}
            except Exception:
                conf = 75.0
                scores = {pred: 75.0}
            return {
                'prediction': pred,
                'confidence': conf,
                'all_scores': scores,
                'input_text': text,
            }
        except Exception as e:
            return {'error': str(e)}


# ==========================================================
#  Report generation
# ==========================================================
def create_report_text(patient_name, patient_id, patient_age, patient_gender,
                       image_result, symptom_result):
    now = datetime.now().strftime("%B %d, %Y  |  %I:%M %p")
    border = "=" * 56

    lines = [
        border,
        "            N E U R O S C A N   A I",
        "       Brain Tumor Detection & Analysis Report",
        border,
        "",
        f"  Date / Time      : {now}",
        f"  Patient Name     : {patient_name or 'N/A'}",
        f"  Patient ID       : {patient_id or 'N/A'}",
        f"  Age              : {patient_age}",
        f"  Gender           : {patient_gender}",
        "",
        "-" * 56,
        "  DIAGNOSTIC RESULTS",
        "-" * 56,
    ]

    if image_result:
        pred = display_name(image_result['prediction'])
        conf = image_result['confidence']
        lines += [
            "",
            "  [MRI Image Analysis - MobileNetV2 CNN]",
            f"    Prediction  : {pred}",
            f"    Confidence  : {conf:.1f}%",
        ]
        if 'all_predictions' in image_result:
            lines.append("    Class-wise breakdown:")
            for label, score in image_result['all_predictions'].items():
                bar = "#" * max(int(score / 4), 0)
                lines.append(f"      {display_name(label):14s} {score:6.1f}%  {bar}")
    else:
        lines += ["", "  [MRI Image Analysis] : Not performed"]

    if symptom_result and 'error' not in symptom_result:
        pred = display_name(symptom_result['prediction'])
        conf = symptom_result.get('confidence', 0)
        lines += [
            "",
            "  [Symptom Analysis - TF-IDF + Logistic Regression]",
            f"    Prediction  : {pred}",
            f"    Confidence  : {conf:.1f}%",
            f"    Symptoms    : \"{symptom_result.get('input_text', '')}\"",
        ]
        if 'all_scores' in symptom_result:
            lines.append("    Class-wise breakdown:")
            for label, score in symptom_result['all_scores'].items():
                bar = "#" * max(int(score / 4), 0)
                lines.append(f"      {display_name(label):14s} {score:6.1f}%  {bar}")
    else:
        lines += ["", "  [Symptom Analysis] : Not performed"]

    if image_result and symptom_result and 'error' not in symptom_result:
        lines += [
            "",
            "-" * 56,
            "  COMBINED ASSESSMENT",
            "-" * 56,
        ]
        ip = image_result['prediction']
        sp = symptom_result['prediction']
        if ip == sp:
            lines.append(f"  Both analyses agree: {display_name(ip)}")
            lines.append("  Recommendation: Results are consistent.")
        else:
            lines.append(f"  MRI suggests: {display_name(ip)}")
            lines.append(f"  Symptoms suggest: {display_name(sp)}")
            lines.append("  Recommendation: Further clinical evaluation advised.")

    lines += [
        "",
        "-" * 56,
        "  DISCLAIMER",
        "-" * 56,
        "  This report is generated by an AI system for research",
        "  and educational purposes ONLY. It does NOT constitute",
        "  a medical diagnosis. Always consult a qualified",
        "  healthcare professional.",
        "",
        border,
        "  NeuroScan AI  |  Powered by MobileNetV2 & Scikit-learn",
        border,
    ]
    return "\n".join(lines)


# ==========================================================
#  Helpers
# ==========================================================
def display_name(prediction):
    if prediction == 'notumor':
        return "No Tumor"
    return prediction.capitalize()


def badge_class(prediction, confidence):
    if prediction == 'notumor':
        return "pred-clear"
    if confidence >= 80:
        return "pred-high"
    if confidence >= 60:
        return "pred-medium"
    return "pred-low"


def render_score_bars(all_predictions):
    html = ""
    for label, score in all_predictions.items():
        html += f"""
        <div class="score-row">
            <span class="score-label">{display_name(label)}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{max(score,1)}%"></div>
            </div>
            <span class="score-value">{score:.1f}%</span>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
#  MAIN
# ==========================================================
def main():
    # Init database on first run
    get_db().close()

    analyzer = BrainTumorAnalyzer()

    # ── Banner ──
    st.markdown("""
    <div class="ns-banner">
        <h1>🧠 NeuroScan AI</h1>
        <p>Intelligent Brain Tumor Detection & Symptom Analysis System</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load CNN ──
    cnn_model = analyzer.load_cnn_model()
    if cnn_model is None:
        st.error(
            "Cannot proceed without CNN model. "
            "Place 'my_brain_tumor_mobilenetv2.h5' in the project root."
        )
        st.stop()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### Patient Information")
        patient_id = st.text_input("Patient ID", key="patient_id")
        patient_name = st.text_input("Patient Name", key="patient_name")
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.markdown("---")
        st.markdown("### Settings")
        show_details = st.checkbox("Show detailed confidence scores", True)

    # ── Navigation ──
    tab_analysis, tab_history = st.tabs(["Analysis", "Patient History"])

    # ============================================================
    #  TAB 1 — ANALYSIS
    # ============================================================
    with tab_analysis:
        col1, col2 = st.columns(2, gap="large")

        # ── MRI IMAGE ANALYSIS ──
        with col1:
            st.markdown(
                '<div class="section-label">MRI Image Analysis</div>',
                unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload MRI scan", type=['png', 'jpg', 'jpeg'],
                help="Supported: PNG, JPG, JPEG (max 10 MB)")

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded MRI Scan", width=280)
                try:
                    with st.spinner("Analyzing MRI scan..."):
                        processed = analyzer.preprocess_image(uploaded_file)
                        results = analyzer.analyze_image(processed, cnn_model)
                        st.session_state.current_mri = results

                        bc = badge_class(results['prediction'],
                                         results['confidence'])
                        st.markdown(f"""
                        <div class="pred-badge {bc}">
                            Prediction: {display_name(results['prediction'])}
                            &mdash; Confidence: {results['confidence']:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

                        if show_details:
                            st.markdown("**Class-wise Scores**")
                            render_score_bars(results['all_predictions'])
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── SYMPTOM ANALYSIS ──
        with col2:
            st.markdown(
                '<div class="section-label">Symptom Analysis</div>',
                unsafe_allow_html=True)

            symptoms_input = st.text_area(
                "Describe symptoms", height=130,
                placeholder="e.g., persistent headaches, blurred vision, nausea...")

            if st.button("Analyze Symptoms", key="btn_sym"):
                if not symptoms_input.strip():
                    st.warning("Please enter symptoms to analyze.")
                else:
                    with st.spinner("Analyzing symptoms..."):
                        result = analyzer.predict_symptoms(symptoms_input)
                        if 'error' in result:
                            st.error(result['error'])
                        else:
                            st.session_state.current_sym = result

            # Always show symptom result from session (survives reruns)
            if 'current_sym' in st.session_state:
                sr = st.session_state.current_sym
                if 'error' not in sr:
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Symptom-Based Analysis</h4>
                        <p><b>Result:</b> {display_name(sr['prediction'])}</p>
                        <p><b>Confidence:</b> {sr['confidence']:.1f}%</p>
                        <p><b>Symptoms:</b> {sr.get('input_text', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if show_details and 'all_scores' in sr:
                        render_score_bars(sr['all_scores'])

        # ── GENERATE REPORT (this is the only action button) ──
        st.markdown("---")
        st.markdown(
            '<div class="section-label">Generate & Save Report</div>',
            unsafe_allow_html=True)

        mri_data = st.session_state.get('current_mri')
        sym_data = st.session_state.get('current_sym')

        if mri_data or sym_data:
            # Show what will be included
            parts = []
            if mri_data:
                parts.append(
                    f"MRI: <b>{display_name(mri_data['prediction'])}</b>"
                    f" ({mri_data['confidence']:.1f}%)")
            if sym_data and 'error' not in sym_data:
                parts.append(
                    f"Symptoms: <b>{display_name(sym_data['prediction'])}</b>"
                    f" ({sym_data.get('confidence',0):.1f}%)")

            st.markdown(f"""
            <div class="result-card">
                <h4>Report will include:</h4>
                <p>{'&nbsp; | &nbsp;'.join(parts)}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Generate Report & Save to Records",
                         key="gen_save", type="primary"):
                if not patient_id.strip() or not patient_name.strip():
                    st.error("Enter Patient ID and Patient Name in the sidebar first.")
                else:
                    report = create_report_text(
                        patient_name=patient_name,
                        patient_id=patient_id,
                        patient_age=patient_age,
                        patient_gender=patient_gender,
                        image_result=mri_data,
                        symptom_result=sym_data,
                    )
                    # Save to database
                    save_record(
                        patient_id=patient_id,
                        patient_name=patient_name,
                        age=patient_age,
                        gender=patient_gender,
                        mri_data=mri_data,
                        symptom_data=sym_data if (sym_data and 'error' not in sym_data) else None,
                        report_text=report,
                    )
                    st.session_state.generated_report = report
                    st.success(
                        f"Report generated and saved for {patient_name} "
                        f"(ID: {patient_id})")

            # Show generated report
            if 'generated_report' in st.session_state:
                st.markdown(
                    f'<div class="report-preview">'
                    f'{st.session_state.generated_report}</div>',
                    unsafe_allow_html=True)
                st.markdown("")
                safe = (patient_name or "patient").replace(" ", "_")
                st.download_button(
                    label="Download Report (.txt)",
                    data=st.session_state.generated_report,
                    file_name=(
                        f"NeuroScan_{safe}"
                        f"_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"),
                    mime="text/plain")

            # Clear button to start fresh analysis
            st.markdown("")
            if st.button("Clear current analysis (start new patient)"):
                for k in ('current_mri', 'current_sym', 'generated_report'):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.caption(
                "Upload an MRI scan or enter symptoms above, "
                "then generate a report here.")

    # ============================================================
    #  TAB 2 — PATIENT HISTORY (from database)
    # ============================================================
    with tab_history:
        st.markdown(
            '<div class="section-label">Patient History</div>',
            unsafe_allow_html=True)

        st.caption(
            "All saved records are stored in the database. "
            "Search by Patient ID or Name to find past reports.")

        # Search controls
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            search_pid = st.text_input(
                "Search by Patient ID",
                placeholder="e.g., 237",
                key="hist_search_id")
        with h_col2:
            search_pname = st.text_input(
                "Search by Patient Name",
                placeholder="e.g., John",
                key="hist_search_name")

        # Quick select from existing patients
        known_patients = get_all_patient_ids()
        if known_patients:
            options = ["All patients"] + [
                f"{pid} - {pname}" for pid, pname in known_patients
            ]
            selected = st.selectbox("Or select a patient:", options,
                                    key="hist_select")
            if selected != "All patients":
                search_pid = selected.split(" - ")[0]

        # Fetch records
        records = search_records(search_pid, search_pname)

        if records:
            st.markdown(f"**Found {len(records)} record(s)**")

            for rec in records:
                rec_id = rec['id']
                ts_raw = rec['created_at'][:19]
                try:
                    ts = datetime.fromisoformat(ts_raw).strftime(
                        "%b %d, %Y  %I:%M %p")
                except Exception:
                    ts = ts_raw

                # Build summary
                parts = []
                if rec['mri_result']:
                    parts.append(
                        f"MRI: <b>{display_name(rec['mri_result'])}</b>"
                        f" ({rec['mri_confidence']:.1f}%)")
                if rec['symptom_result']:
                    parts.append(
                        f"Symptoms: <b>{display_name(rec['symptom_result'])}</b>"
                        f" ({rec['symptom_confidence']:.1f}%)")
                body = " &nbsp;|&nbsp; ".join(parts) if parts else "No data"

                st.markdown(f"""
                <div class="history-card">
                    <div class="hc-title">
                        {rec['patient_name']} (ID: {rec['patient_id']})
                        &nbsp;&middot;&nbsp; {ts}
                        &nbsp;&middot;&nbsp; Age: {rec['age']}, {rec['gender']}
                    </div>
                    <div class="hc-body">{body}</div>
                </div>
                """, unsafe_allow_html=True)

                # Action buttons
                b1, b2, b3 = st.columns([1, 1, 4])
                with b1:
                    if st.button("View Report", key=f"view_{rec_id}"):
                        st.session_state[f"show_{rec_id}"] = not st.session_state.get(
                            f"show_{rec_id}", False)
                        st.rerun()
                with b2:
                    if st.button("Delete", key=f"del_{rec_id}"):
                        delete_record(rec_id)
                        st.success("Record deleted.")
                        st.rerun()

                # Show/hide report
                if st.session_state.get(f"show_{rec_id}", False):
                    report = rec.get('report_text', '')
                    if report:
                        st.markdown(
                            f'<div class="report-preview">{report}</div>',
                            unsafe_allow_html=True)
                        safe = rec['patient_name'].replace(" ", "_")
                        st.download_button(
                            label="Download this report",
                            data=report,
                            file_name=f"NeuroScan_{safe}_{rec_id}.txt",
                            mime="text/plain",
                            key=f"dl_{rec_id}")
                    else:
                        st.info("No report text saved for this record.")
        else:
            if search_pid or search_pname:
                st.info("No records found matching your search.")
            else:
                st.info(
                    "No records in the database yet. "
                    "Generate a report from the Analysis tab to save one.")

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div class="ns-footer">
        NeuroScan AI v3.0 &middot; Powered by MobileNetV2 & Scikit-learn<br>
        For research and educational purposes only.
        Always consult a healthcare professional.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
