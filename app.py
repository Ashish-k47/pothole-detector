
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# -------------------------------
# Configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "pothole_best.pt"

st.set_page_config(
    page_title="Pothole Detection",
    page_icon=None,
    layout="wide",
)

st.title("Pothole Detection System")
st.write(
    "Upload a road image to detect potholes and other "
    "road defects using your trained YOLO model."
)


# -------------------------------
# Load trained YOLO model
# -------------------------------

@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {WEIGHTS_PATH}"
        )

    return YOLO(str(WEIGHTS_PATH))


# -------------------------------
# Image upload
# -------------------------------

uploaded = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp"],
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.90,
    value=0.10,
    step=0.05,
)

# -------------------------------
# Detection
# -------------------------------

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        st.write("Image uploaded successfully.")

        if st.button("Detect Road Defects", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()

                result = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=640,
                    device="cpu",
                    verbose=False,
                )[0]

                boxes = result.boxes

                # Create a copy for drawing detections
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)

                # Font for labels
                try:
                    font = ImageFont.truetype(
                        "arial.ttf", size=max(16, image.width // 60)
                    )
                except OSError:
                    font = ImageFont.load_default()

                # Scale the bounding box thickness
                thickness = max(2, image.width // 300)

                # Draw red boxes and labels
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = [
                            int(value)
                            for value in box.xyxy[0].tolist()
                        ]

                        score = float(box.conf[0].item())
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]

                        label = f"{class_name} {score:.2f}"

                        # Red bounding box
                        for offset in range(thickness):
                            draw.rectangle(
                                (
                                    x1 - offset,
                                    y1 - offset,
                                    x2 + offset,
                                    y2 + offset,
                                ),
                                outline="red",
                            )

                        # Red label text
                        text_y = max(0, y1 - 25)
                        draw.text(
                            (x1, text_y),
                            label,
                            fill="red",
                            font=font,
                            stroke_width=0,
                        )

                else:
                    draw.text(
                        (15, 15),
                        "No trained class detected",
                        fill="red",
                        font=font,
                    )

            # -------------------------------
            # Display images side-by-side
            # -------------------------------

            left, right = st.columns(2, gap="medium")

            with left:
                st.subheader("Original Image")
                st.image(
                    image,
                    use_container_width=True,
                )

            with right:
                st.subheader("Detection Result")
                st.image(
                    annotated,
                    use_container_width=True,
                )

            # -------------------------------
            # Detection summary
            # -------------------------------

            if boxes is None or len(boxes) == 0:
                st.info(
                    "No trained class was detected. "
                    "This does not guarantee the road is free of defects."
                )

            else:
                st.success(
                    f"{len(boxes)} detection(s) found."
                )

                details = []

                for box in boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    details.append(
                        {
                            "Detected Class": model.names[class_id],
                            "Confidence": f"{score:.1%}",
                        }
                    )

                st.dataframe(
                    details,
                    use_container_width=True,
                    hide_index=True,
                )

            # -------------------------------
            # Download annotated image
            # -------------------------------

            buffer = BytesIO()
            annotated.save(buffer, format="JPEG", quality=95)

            st.download_button(
                "Download Detection Result",
                data=buffer.getvalue(),
                file_name="pothole_result.jpg",
                mime="image/jpeg",
            )

            st.caption(
                "Predictions are estimates. A missed detection does not "
                "prove that the road is free of potholes."
            )

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
