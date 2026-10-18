
import streamlit as st
import cv2
import tempfile
import os
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
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
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
                      summary, keywords, flagged):
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

    story.append(Paragraph("CrimeVision AI", styles["Title"]))
    story.append(Paragraph(
        "Surveillance Video Analysis Report",
        styles["Heading2"]
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

    story.append(Paragraph("NLP Observation Summary", styles["Heading2"]))
    story.append(Paragraph(escape(str(summary)), styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Important Keywords", styles["Heading2"]))
    story.append(Paragraph(
        escape(", ".join(keywords) if keywords else "None available"),
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("ML Pattern Review", styles["Heading2"]))
    if flagged:
        story.append(Paragraph(
            f"{len(flagged)} unusual object-count patterns were flagged "
            "for human review.",
            styles["Normal"]
        ))
        for item in flagged:
            seconds = int(item["time"])
            timestamp = f"{seconds // 60:02d}:{seconds % 60:02d}"
            story.append(Paragraph(
                f"{timestamp} — {item['count']} object detections",
                styles["Normal"]
            ))
    else:
        story.append(Paragraph(
            "No unusual patterns were flagged, or no flags were available.",
            styles["Normal"]
        ))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "<b>Important:</b> This report contains AI-generated visual "
        "observations and statistical anomaly flags. It does not establish "
        "that a crime occurred or identify a person as a criminal. "
        "Human review is required.",
        styles["Normal"]
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- WEBSITE ----------
st.set_page_config(
    page_title="CrimeVision AI",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111827);
    color: #f8fafc;
}
h1, h2, h3, p, label {
    color: #f8fafc;
}
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎥 CrimeVision AI")
st.write(
    "AI-Based Automated Crime Scene and Surveillance Video Report Generator"
)
st.caption("AI-assisted visual analysis. Human review is required.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


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
                            "objects": detected,
                            "frame": result.plot()
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

                if observations:
                    st.image(
                        cv2.cvtColor(
                            observations[0]["frame"],
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="YOLO object detection",
                        use_container_width=True
                    )

                st.subheader("Timestamped Observations")
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

                    st.write(f"**{timestamp}** — {objects}")

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
                    observations=[],
                    summary=summary,
                    keywords=keywords,
                    flagged=[]
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
