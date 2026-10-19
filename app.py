
import streamlit as st
import cv2
import tempfile
import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)
from collections import Counter
from ultralytics import YOLO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from xml.sax.saxutils import escape

# ---------- NLP ----------
def generate_nlp_summary(observations):
    sentences = []

    for item in observations:
        objects = item["objects"]

        if objects:
            seconds = int(item["time"])
            timestamp = f"{seconds // 60:02d}:{seconds % 60:02d}"
            sentence = (
                f"At {timestamp}, detected objects included "
                + ", ".join(sorted(set(objects))) + "."
            )
            sentences.append(sentence)

    if not sentences:
        return "No objects were detected.", []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(sentences)
        terms = vectorizer.get_feature_names_out()
        scores = matrix.sum(axis=0).A1

        keywords = sorted(
            zip(terms, scores),
            key=lambda x: x[1],
            reverse=True
        )[:8]

        sentence_scores = matrix.sum(axis=1).A1
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: sentence_scores[i],
            reverse=True
        )

        selected = sorted(ranked[:3])
        summary = " ".join(sentences[i] for i in selected)

        return summary, [word for word, score in keywords]

    except ValueError:
        return "Not enough text to generate a summary.", []

def review_unusual_patterns(observations):
    """Flag unusual object-count patterns for human review."""

    if len(observations) < 10:
        return (
            [],
            "At least 10 analyzed frames are needed for ML review."
        )

    # Count detections in each sampled frame
    frame_counts = [
        len(item["objects"]) for item in observations
    ]

    # ML works on the number of detections per frame
    data = [[count] for count in frame_counts]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42
    )

    predictions = model.fit_predict(data)

    flagged = []

    for index, prediction in enumerate(predictions):
        if prediction == -1:
            flagged.append({
                "time": observations[index]["time"],
                "count": frame_counts[index]
            })

    return flagged, None


def create_pdf_report(video_name, object_counts, observations,
                      summary, keywords, flagged,
                      fight_prediction=None, fight_scores=None):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

        # ---------- PROFESSIONAL COVER PAGE ----------
    title_style = ParagraphStyle(
        "CrimeVisionTitle",
        parent=styles["Title"],
        fontSize=26,
        leading=32,
        textColor=colors.HexColor("#17365D"),
        alignment=1,
        spaceAfter=16,
    )

    subtitle_style = ParagraphStyle(
        "CrimeVisionSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#2874A6"),
        alignment=1,
    )

    story.append(Spacer(1, 95))
    story.append(Paragraph("CRIMEVISION AI", title_style))
    story.append(Paragraph(
        "Surveillance Video Analysis Report",
        subtitle_style
    ))
    story.append(Spacer(1, 30))

    cover_prediction = (
        str(fight_prediction).replace("_", " ").title()
        if fight_prediction else "Not available"
    )

    cover_data = [
        ["REPORT ITEM", "DETAILS"],
        ["Video file", escape(str(video_name))],
        ["Frames analyzed", str(len(observations))],
        ["Predicted class", escape(cover_prediction)],
        [
            "Fight score",
            f"{fight_scores.get('fight', 0):.2f}%"
            if fight_scores else "N/A"
        ],
        [
            "Non-fight score",
            f"{fight_scores.get('non_fight', 0):.2f}%"
            if fight_scores else "N/A"
        ],
    ]

    cover_table = Table(
        cover_data,
        colWidths=[140, 260],
        repeatRows=1,
        hAlign="CENTER"
    )

    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1),
         colors.HexColor("#EAF0F6")),
        ("GRID", (0, 0), (-1, -1), 0.5,
         colors.HexColor("#CCD6E0")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    story.append(cover_table)
    story.append(Spacer(1, 30))

    story.append(Paragraph(
        "Experimental AI results require human review. "
        "This report does not establish that a crime occurred.",
        styles["Normal"]
    ))

    story.append(PageBreak())

    # ---------- DETAILED REPORT ----------
    story.append(Paragraph(
        "Detailed Analysis",
        styles["Heading1"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Video:</b> {escape(str(video_name))}",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Frames analyzed:</b> {len(observations)}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detected Object Counts", styles["Heading2"]))

    table_data = [["Object Category", "Detection Count"]]
    for name, count in object_counts.most_common():
        table_data.append([escape(str(name)), str(count)])

    if len(table_data) == 1:
        table_data.append(["No objects detected", "0"])

    table = Table(table_data, colWidths=[250, 150], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    
    # ---------- NLP OBSERVATION SUMMARY ----------
    story.append(Paragraph(
        "NLP Observation Summary",
        styles["Heading2"]
    ))

    summary_style = ParagraphStyle(
        "SummaryStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )

    story.append(Paragraph(
        escape(str(summary)),
        summary_style
    ))
    story.append(Spacer(1, 10))

    # ---------- IMPORTANT KEYWORDS ----------
    story.append(Paragraph(
        "Important Keywords",
        styles["Heading2"]
    ))

    keyword_text = ", ".join(keywords) if keywords else "None available"

    story.append(Paragraph(
        escape(keyword_text),
        summary_style
    ))
    story.append(Spacer(1, 12))

    # ---------- ML PATTERN REVIEW ----------
    story.append(Paragraph(
        "ML Pattern Review",
        styles["Heading2"]
    ))

    if flagged:
        story.append(Paragraph(
            f"{len(flagged)} unusual object-count patterns "
            "were flagged for human review.",
            summary_style
        ))

        pattern_data = [
            ["Timestamp", "Object Detection Count"]
        ]

        for item in flagged:
            seconds = int(item["time"])
            timestamp = (
                f"{seconds // 3600:02d}:"
                f"{(seconds % 3600) // 60:02d}:"
                f"{seconds % 60:02d}"
            )

            pattern_data.append([
                timestamp,
                str(item["count"])
            ])

        pattern_table = Table(
            pattern_data,
            colWidths=[130, 220],
            repeatRows=1,
            hAlign="LEFT"
        )

        pattern_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.HexColor("#CCD6E0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))

        story.append(pattern_table)

    else:
        story.append(Paragraph(
            "No unusual patterns were flagged, or no flags were available.",
            summary_style
        ))

    story.append(Spacer(1, 16))

    # ---------- RESPONSIBLE USE DISCLAIMER ----------
    story.append(Paragraph(
        "Important: This report contains AI-generated visual "
        "observations and statistical anomaly flags. It does not establish "
        "that a crime occurred or identify a person as a criminal. "
        "Human review is required.",
        summary_style
    ))
    story.append(Spacer(1, 12))


    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- WEBSITE ----------
st.set_page_config(
    page_title="CrimeVision AI",
    page_icon="🎥",
    layout="wide"
)
# ---------- LOGIN AND GUEST ACCESS ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False

if not st.session_state.authenticated and not st.session_state.guest_mode:

    
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #16072e, #2e1065, #4c1d95
        ) !important;
    }

    .login-title {
        text-align: center;
        color: #c4b5fd;
        font-size: 38px;
        font-weight: 800;
    }

    .login-subtitle {
        text-align: center;
        color: #ddd6fe;
        margin-bottom: 25px;
    }

    h1, h2, h3, p, label,
    [data-testid="stMarkdownContainer"] {
        color: #ede9fe;
    }

    div[data-testid="stButton"] > button {
        background: #8b5cf6 !important;
        background-color: #8b5cf6 !important;
        color: white !important;
        border: 2px solid #c4b5fd !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 46px !important;
    }

    div[data-testid="stButton"] > button:hover {
        background: #c4b5fd !important;
        color: #2e1065 !important;
    }

    .stTextInput input {
        background-color: #241044 !important;
        color: white !important;
        border: 1px solid #8b5cf6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="login-title">🎥 CrimeVision AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="login-subtitle">'
        'Intelligent Video Analytics & Evidence Reporting'
        '</div>',
        unsafe_allow_html=True
    )

    _, login_col, _ = st.columns([1, 1.2, 1])

    with login_col:
        st.subheader("Welcome back")

        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            login_email = st.text_input(
                "Email address", key="login_email"
            )
            login_password = st.text_input(
                "Password", type="password", key="login_password"
            )

            if st.button("Login", use_container_width=True):
                st.warning(
                    "Account login will work after we connect "
                    "the registration and authentication system."
                )

        with register_tab:
            register_email = st.text_input(
                "Email address", key="register_email"
            )
            register_password = st.text_input(
                "Create password", type="password",
                key="register_password"
            )

            if st.button("Create account", use_container_width=True):
                st.info(
                    "Account registration will be connected in "
                    "the next step."
                )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;color:#75658e;'>"
            "Or continue without an account"
            "</div>",
            unsafe_allow_html=True
        )

        if st.button("Continue as Guest", use_container_width=True):
            st.session_state.guest_mode = True
            st.rerun()

    st.stop()

# ---------- END LOGIN SCREEN ----------



st.markdown("""
<style>
/* Dark purple and light purple theme */
.stApp {
    background: linear-gradient(135deg, #16072e 0%, #2e1065 50%, #4c1d95 100%) !important;
    color: #ede9fe !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

.block-container {
    padding-top: 3rem;
}

/* Title and subtitle */
.login-title {
    text-align: center;
    color: #c4b5fd !important;
    font-size: 38px;
    font-weight: 800;
}

.login-subtitle {
    text-align: center;
    color: #ddd6fe !important;
    margin-bottom: 25px;
}

/* Headings and labels */
h1, h2, h3, p, label,
[data-testid="stMarkdownContainer"] {
    color: #ede9fe;
}

/* Purple buttons */
.stButton > button {
    background: linear-gradient(110deg, #7c3aed, #a78bfa) !important;
    color: #1e0a3c !important;
    border: 1px solid #c4b5fd !important;
    border-radius: 12px !important;
    min-height: 46px;
    font-weight: 800 !important;
    box-shadow: 0 5px 18px rgba(167, 139, 250, 0.2);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(110deg, #a78bfa, #c4b5fd) !important;
    color: #16072e !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(196, 181, 253, 0.3);
}

/* Dark input fields */
.stTextInput input {
    background-color: #241044 !important;
    color: #f5f3ff !important;
    border: 1px solid #8b5cf6 !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder {
    color: #c4b5fd !important;
}

.stTextInput input:focus {
    border: 2px solid #c4b5fd !important;
    box-shadow: 0 0 0 2px #6d28d9 !important;
}

/* Login and Register tabs */
.stTabs [data-baseweb="tab"] {
    color: #c4b5fd !important;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #c4b5fd !important;
}

/* Divider and cards */
hr {
    border-color: #6d28d9 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(46, 16, 101, 0.45) !important;
    border: 1px solid #6d28d9 !important;
    border-radius: 18px !important;
}

/* Guest access text */
[data-testid="stMarkdownContainer"] div {
    color: #ddd6fe;
}

/* File uploader, if visible on this page */
div[data-testid="stFileUploader"] {
    background: #241044;
    border-color: #8b5cf6;
    color: #ede9fe;
}
</style>
""", unsafe_allow_html=True)


st.caption(
    "AI-assisted visual analysis · Results require human review "
    "and do not establish that a crime occurred."
)


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")
@st.cache_resource
def load_fighting_model():
    model_dir = "models/fighting_videomae"

    processor = VideoMAEImageProcessor.from_pretrained(model_dir)
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)

    model.eval()
    print("FIGHT MODEL LABELS:", model.config.id2label)
    return processor, model

def predict_fight_video(video_path, processor, model):
    device = next(model.parameters()).device

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video for fight analysis.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("No readable video frames found.")

    positions = np.linspace(
        0, total_frames - 1, 16
    ).astype(int)

    frames = []

    try:
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            success, frame = cap.read()

            if not success:
                raise RuntimeError("Could not read a sampled frame.")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()

    inputs = processor(images=frames, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"].to(device)
        )
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    results = {}

    for index, probability in enumerate(probabilities):
        label = model.config.id2label[index]
        results[label] = float(probability.item() * 100)

    predicted_label = max(results, key=results.get)

    return predicted_label, results

def detect_fight_moments(video_path, processor, model,
                         segment_seconds=4, threshold=60):
    """Scan short video segments and flag possible fight moments."""

    device = next(model.parameters()).device
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not read video duration or FPS.")

    duration = total_frames / fps
    moments = []

    try:
        start_time = 0.0

        while start_time < duration:
            end_time = min(start_time + segment_seconds, duration)
            frames = []

            # Sample 16 frames from this segment
            positions = np.linspace(
                int(start_time * fps),
                max(int(start_time * fps), int(end_time * fps) - 1),
                16
            ).astype(int)

            for position in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                success, frame = cap.read()

                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))

            if len(frames) == 16:
                inputs = processor(images=frames, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(
                        pixel_values=inputs["pixel_values"].to(device)
                    )
                    probabilities = torch.softmax(
                        outputs.logits, dim=-1
                    )[0]

                scores = {
                    model.config.id2label[i]:
                    float(probabilities[i].item() * 100)
                    for i in range(len(probabilities))
                }
                print("Model class labels:", model.config.id2label)
                print("Segment scores:", scores)
               
                fight_labels = [
                    label for label in scores
                    if "fight" in label.lower()
                    and "non" not in label.lower()
                ]

                fight_score = (
                    scores[fight_labels[0]]
                    if fight_labels
                    else 0.0
                )

                
                print("Fight score used:", fight_score)

                if fight_score >= threshold:
                    moments.append({
                        "start": start_time,
                        "end": end_time,
                        "fight_score": fight_score
                    })


                

            start_time = end_time

    finally:
        cap.release()

    return moments


st.sidebar.title("AI Technologies")
st.sidebar.write("🖼️ DIP — OpenCV")
st.sidebar.write("🧠 NN — YOLO11n")
st.sidebar.write("📊 ML — Planned")
st.sidebar.write("📝 NLP — TF-IDF summaries")

st.subheader("Upload Surveillance Video")

video = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov", "mkv"]
)

if video is not None:
    st.video(video)

    if st.button("Start AI Analysis", type="primary"):
        temporary_path = None
        cap = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(video.name)[1]
            ) as temp:
                temp.write(video.getbuffer())
                temporary_path = temp.name

            with st.spinner("Loading YOLO model..."):
                model = load_model()
            with st.spinner("Loading fight detection model..."):
                fighting_processor, fighting_model = load_fighting_model()
            cap = cv2.VideoCapture(temporary_path)

            if not cap.isOpened():
                st.error("Could not open this video. Try another file.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                if fps <= 0:
                    fps = 25

                st.info(f"Total video frames: {total_frames}")

                progress = st.progress(0)
                status = st.empty()

                object_counts = Counter()
                observations = []
                frame_number = 0
                analyzed = 0
                max_frames = 100
                sample_every = max(1, total_frames // max_frames)

                while analyzed < max_frames:
                    success, frame = cap.read()

                    if not success:
                        break

                    if frame_number % sample_every == 0:
                        # DIP: resize video frame
                        height, width = frame.shape[:2]

                        if width > 640:
                            new_height = int(height * 640 / width)
                            frame = cv2.resize(
                                frame, (640, new_height)
                            )

                        # NN: YOLO object detection
                        result = model.predict(
                            frame,
                            conf=0.35,
                            verbose=False
                        )[0]

                        detected = []

                        for class_id in result.boxes.cls.tolist():
                            name = result.names[int(class_id)]
                            object_counts[name] += 1
                            detected.append(name)

                        observations.append({
                            "time": frame_number / fps,
                            "objects": detected
                        })

                        analyzed += 1

                        if total_frames > 0:
                            progress.progress(
                                min(frame_number / total_frames, 1.0)
                            )

                        status.write(
                            f"Processed {analyzed} sampled frames"
                        )

                    frame_number += 1

                cap.release()
                cap = None

                progress.progress(1.0)
                status.success("Video analysis completed!")
                
                # VideoMAE: Fight / Non-fight prediction
                st.subheader("🥊 Fight / Non-Fight Analysis")

                with st.spinner("Analyzing video for fight-related actions..."):
                    predicted_label, fight_scores = predict_fight_video(
                        temporary_path,
                        fighting_processor,
                        fighting_model
                    )
                st.write("Debug — model class labels:", fighting_model.config.id2label)
                st.write("Debug — raw fight scores:", fight_scores)
                # Detect possible fight moments in short segments
                st.subheader("⏱️ Possible Fighting Moments")

                with st.spinner("Scanning video segments for possible fighting moments..."):
                    fight_moments = detect_fight_moments(
                        temporary_path,
                        fighting_processor,
                        fighting_model,
                        segment_seconds=4,
                        threshold=60
                    )

                if fight_moments:
                    for moment in fight_moments:
                        start = moment["start"]
                        end = moment["end"]

                        def format_timestamp(seconds):
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            secs = seconds % 60
                            return (
                                f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
                            )

                        st.warning(
                            f"Possible activity requiring review: "
                            f"{format_timestamp(start)} – "
                            f"{format_timestamp(end)} "
                            f"(fight score: {moment['fight_score']:.1f}%)"
                        )
                else:
                    st.info(
                        "No segments crossed the experimental threshold. "
                        "This does not guarantee that no fighting occurred."
                    )

                st.metric(
                    "Predicted Class",
                    predicted_label.replace("_", " ").title()
                )

                score_col1, score_col2 = st.columns(2)

                fight_label = next(
                    (
                        label for label in fight_scores
                        if "fight" in label.lower()
                        and "non" not in label.lower()
                    ),
                    None
                )

                non_fight_label = next(
                    (
                        label for label in fight_scores
                        if "non" in label.lower()
                        and "fight" in label.lower()
                    ),
                    None
                )

                score_col1.metric(
                    "Fight Score",
                    f"{fight_scores[fight_label]:.2f}%"
                    if fight_label else "Unknown label"
                )

                score_col2.metric(
                    "Non-Fight Score",
                    f"{fight_scores[non_fight_label]:.2f}%"
                    if non_fight_label else "Unknown label"
                )

                

                st.warning(
                    "Experimental AI prediction only. "
                    "This result does not prove that a crime occurred. "
                    "Human review is required."
                )

                # Results
                st.subheader("Analysis Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Frames Analyzed", analyzed)
                col2.metric("Object Categories", len(object_counts))
                col3.metric(
                    "Total Detections",
                    sum(object_counts.values())
                )

                st.subheader("Detected Objects")

                if object_counts:
                    st.bar_chart(dict(object_counts))
                else:
                    st.warning("No objects were detected.")

                st.subheader("Sample Analyzed Frame")

                st.info(
                    "Frame-level object detections are included in the "
                    "timestamped observations. The annotated frame preview "
                    "is disabled to reduce memory usage."
                )

                
                st.subheader("Timestamped Observations")

                observation_rows = []

                for item in observations[:20]:
                    total_seconds = max(0, float(item["time"]))

                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = total_seconds % 60

                    timestamp = (
                        f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
                    )

                    objects = (
                        ", ".join(item["objects"])
                        if item["objects"]
                        else "No objects detected"
                    )

                    observation_rows.append({
                        "Timestamp": timestamp,
                        "Detected Objects": objects
                    })

                if observation_rows:
                    st.dataframe(
                        observation_rows,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No observations available.")

                # NLP
                st.divider()
                st.subheader("📝 NLP Analysis")

                summary, keywords = generate_nlp_summary(observations)

                st.markdown("### Automatic Observation Summary")
                st.write(summary)

                st.markdown("### Important Keywords")

                if keywords:
                    st.write(" • ".join(keywords))
                else:
                    st.write("No keywords available.")
            
                st.caption(
                    "Visual detections are observations only; "
                    "they do not establish that a crime occurred."
                )
                
                # ML: unusual object-count pattern review
                st.divider()
                st.subheader("📊 ML Pattern Review")

                flagged, ml_message = review_unusual_patterns(
                    observations
                )

                if ml_message:
                    st.info(ml_message)
                elif flagged:
                    st.warning(
                        f"{len(flagged)} unusual object-count "
                        "patterns flagged for human review."
                    )

                    for item in flagged:
                        seconds = int(item["time"])
                        timestamp = (
                            f"{seconds // 60:02d}:"
                            f"{seconds % 60:02d}"
                        )
                        st.write(
                            f"**{timestamp}** — "
                            f"{item['count']} object detections "
                            "in this sampled frame."
                        )
                else:
                    st.success(
                        "No unusual object-count patterns "
                        "were flagged by this model."
                    )
                    
                st.divider()
                st.subheader("📄 Download Analysis Report")
                flagged, ml_message = review_unusual_patterns(observations)
                
                
                pdf_data = create_pdf_report(
                       video_name=video.name,
                       object_counts=object_counts,
                       observations=observations,
                       summary=summary,
                       keywords=keywords,
                       flagged=flagged,
                       fight_prediction=predicted_label,
                       fight_scores=fight_scores
                )



                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_data,
                    file_name="CrimeVision_AI_Report.pdf",
                    mime="application/pdf",
                    type="primary"
                )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

        finally:
            if cap is not None:
                cap.release()

            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)
