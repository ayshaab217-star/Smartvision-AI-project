import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartVision",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

YOLO_PATH = r"D:\Smartvision_project\models\smartvision_yolov8n_25class_best.pt"
CNN_PATH = r"D:\Smartvision_project\models\efficientnetb0_final.keras"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "couch",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "truck"
]


# ============================================================
# LOAD YOLO
# ============================================================

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_PATH)


# ============================================================
# LOAD CNN
# ============================================================

@st.cache_resource
def load_cnn():
    return load_model(CNN_PATH)


try:
    yolo_model = load_yolo()
    cnn_model = load_cnn()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error("❌ Could not load the models.")
    st.code(str(e))


# ============================================================
# HEADER
# ============================================================

st.title("🔍 SmartVision")

st.subheader(
    "AI-Powered Object Detection & Verification"
)

st.write(
    "Upload an image to detect objects using YOLO "
    "and verify predictions using EfficientNetB0."
)


if models_loaded:

    st.success(
        "✅ YOLO + EfficientNetB0 loaded successfully"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence_threshold = st.slider(
        "YOLO Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.info(
        "Objects below this confidence level "
        "will not be displayed."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None and models_loaded:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # --------------------------------------------------------
    # DISPLAY ORIGINAL IMAGE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    with st.spinner("🔍 Detecting objects..."):

        start_time = time.time()

        results = yolo_model.predict(
            source=np.array(image),
            conf=confidence_threshold,
            verbose=False
        )

        yolo_time = time.time() - start_time


    result = results[0]


    # --------------------------------------------------------
    # ANNOTATED IMAGE
    # --------------------------------------------------------

    with col2:

        st.subheader("🎯 Detection Image")

        annotated_image = result.plot()

        st.image(
            annotated_image,
            use_container_width=True
        )


    st.divider()


    # ========================================================
    # DETECTION RESULTS
    # ========================================================

    st.header("🎯 SmartVision Results")


    if len(result.boxes) == 0:

        st.warning(
            "No objects detected at the selected "
            "confidence threshold."
        )

    else:

        st.success(
            f"Objects detected: {len(result.boxes)}"
        )


        # ----------------------------------------------------
        # PROCESS EACH DETECTION
        # ----------------------------------------------------

        final_results = []


        for object_number, box in enumerate(
            result.boxes,
            start=1
        ):

            # ----------------------------------------------
            # YOLO INFORMATION
            # ----------------------------------------------

            class_id = int(box.cls[0])

            yolo_confidence = float(
                box.conf[0]
            )

            yolo_class = yolo_model.names[
                class_id
            ]


            # ----------------------------------------------
            # BOUNDING BOX
            # ----------------------------------------------

            coordinates = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = coordinates

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.width, int(x2))
            y2 = min(image.height, int(y2))


            # ----------------------------------------------
            # CROP OBJECT FOR CNN
            # ----------------------------------------------

            if x2 > x1 and y2 > y1:

                cropped_object = image.crop(
                    (x1, y1, x2, y2)
                )

                cnn_image = cropped_object.resize(
                    (224, 224)
                )

                cnn_array = np.array(
                    cnn_image
                ).astype("float32")

                cnn_array = np.expand_dims(
                    cnn_array,
                    axis=0
                )


                # ------------------------------------------
                # CNN PREDICTION
                # ------------------------------------------

                cnn_start = time.time()

                cnn_prediction = cnn_model.predict(
                    cnn_array,
                    verbose=0
                )

                cnn_time = (
                    time.time() - cnn_start
                )


                cnn_class_id = int(
                    np.argmax(cnn_prediction[0])
                )

                cnn_confidence = float(
                    cnn_prediction[0][cnn_class_id]
                )

                cnn_class = CLASS_NAMES[
                    cnn_class_id
                ]

            else:

                cnn_class = "Unknown"

                cnn_confidence = 0.0

                cnn_time = 0


            # ----------------------------------------------
            # VERIFICATION
            # ----------------------------------------------

            if yolo_class == cnn_class:

                status = "✅ VERIFIED"

            else:

                status = "⚠️ REVIEW"


            # ----------------------------------------------
            # STORE RESULT
            # ----------------------------------------------

            final_results.append({

                "Object": object_number,

                "YOLO Class": yolo_class,

                "YOLO Confidence": yolo_confidence,

                "CNN Prediction": cnn_class,

                "CNN Confidence": cnn_confidence,

                "Status": status

            })


        # ====================================================
        # RESULTS TABLE
        # ====================================================

        st.subheader("📊 Object Verification")

        for item in final_results:

            st.markdown(
                f"""
                ### Object {item["Object"]}

                **YOLO Class:** {item["YOLO Class"]}

                **YOLO Confidence:** {item["YOLO Confidence"]:.2f}

                **CNN Prediction:** {item["CNN Prediction"]}

                **CNN Confidence:** {item["CNN Confidence"]:.2f}

                **Status:** {item["Status"]}
                """
            )

            st.divider()


        # ====================================================
        # SUMMARY TABLE
        # ====================================================

        st.subheader("📋 Final Results")

        table_data = []

        for item in final_results:

            table_data.append({

                "Object": item["Object"],

                "YOLO": item["YOLO Class"],

                "YOLO Confidence":
                    f'{item["YOLO Confidence"]:.2f}',

                "CNN": item["CNN Prediction"],

                "CNN Confidence":
                    f'{item["CNN Confidence"]:.2f}',

                "Status": item["Status"]

            })


        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SPEED
        # ====================================================

        st.subheader("⚡ Performance")

        st.write(
            f"YOLO inference time: "
            f"**{yolo_time:.3f} seconds**"
        )

        total_time = (
            yolo_time +
            sum(
                0 for _ in final_results
            )
        )

        st.write(
            "Complete pipeline includes "
            "YOLO detection and CNN verification."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartVision — AI Object Detection & Verification "
    "using YOLOv8 and EfficientNetB0"
)