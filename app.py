# import streamlit as st
# from PIL import Image
# import torch
# import numpy as np
# import cv2
# import io
# import os

# # Load the model once at the start
# @st.cache_resource
# def load_model():
#     model_path = "best.pt"
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#     return model

# # Inference function
# def run_inference(model, image: Image.Image):
#     # Convert PIL image to numpy
#     image_np = np.array(image)

#     # Run inference
#     results = model(image_np)

#     # Render predictions on the image
#     result_img = np.squeeze(results.render())  # returns list with np array

#     # Convert back to PIL Image
#     result_pil = Image.fromarray(result_img)
#     return result_pil

# # UI
# st.set_page_config(page_title="Fracture Detection", layout="centered")
# st.title("🦴 Fracture Detection using Trained Model")

# uploaded_file = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     # Load input image
#     input_image = Image.open(uploaded_file).convert("RGB")
#     st.subheader("Input Image:")
#     st.image(input_image, use_column_width=True)

#     # Load model and run inference
#     with st.spinner("Running model..."):
#         model = load_model()
#         result_image = run_inference(model, input_image)

#     # Show output
#     st.subheader("Output Image:")
#     st.image(result_image, use_column_width=True)

#     # Prepare image for download
#     img_byte_arr = io.BytesIO()
#     result_image.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()

#     st.download_button(
#         label="📥 Download Result Image",
#         data=img_byte_arr,
#         file_name="fracture_detection_result.png",
#         mime="image/png"
#     )


# import streamlit as st
# from PIL import Image
# import torch
# import numpy as np
# import io
# import os
# from ultralytics import YOLO

# # Load the model once
# @st.cache_resource
# def load_model():
#     model = YOLO("best.pt")  # Your custom-trained model
#     return model

# # Run inference
# def run_inference(model, input_image: Image.Image) -> Image.Image:
#     # Convert to numpy
#     img_np = np.array(input_image)

#     # Run model prediction
#     results = model.predict(source=img_np, save=False, conf=0.25)

#     # Render results on image
#     annotated_frame = results[0].plot()

#     # Convert to PIL
#     result_pil = Image.fromarray(annotated_frame)
#     return result_pil

# # Streamlit UI
# st.set_page_config(page_title="Safety Tool Detection", layout="centered")
# st.title("Safety Tool Detection using YOLO Model")

# uploaded_file = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     input_image = Image.open(uploaded_file).convert("RGB")
#     st.subheader("Input Image:")
#     st.image(input_image, use_column_width=True)

#     # Load model and run inference
#     with st.spinner("Running model..."):
#         model = load_model()
#         result_image = run_inference(model, input_image)

#     # Show output
#     st.subheader("Detected Output:")
#     st.image(result_image, use_column_width=True)

#     # Download button
#     buf = io.BytesIO()
#     result_image.save(buf, format="PNG")
#     byte_im = buf.getvalue()

#     st.download_button(
#         label="📥 Download Result Image",
#         data=byte_im,
#         file_name="fracture_result.png",
#         mime="image/png"
#     )


import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import tempfile
import cv2
import os

# Load model only once
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# Inference function
def run_inference(model, input_data, is_video=False):
    if not is_video:
        # --------- IMAGE HANDLING ---------
        img_np = np.array(input_data)
        results = model.predict(source=img_np, save=False, conf=0.25)
        annotated = results[0].plot()
        return Image.fromarray(annotated)

    else:
        # --------- VIDEO HANDLING ---------
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(input_data.read())
        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = "output_result.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model.predict(source=frame, save=False, conf=0.25)
            annotated = results[0].plot()
            out.write(annotated)

        cap.release()
        out.release()
        return out_path

# UI layout
st.set_page_config(page_title="Safety Tool Detection", layout="centered")
st.title("🦺 Safety Tool Detection using YOLOv5/YOLOv8 Model")

uploaded_file = st.file_uploader("Upload an Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    model = load_model()

    # IMAGE
    if "image" in file_type:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Input Image:")
        st.image(input_image, use_column_width=True)

        with st.spinner("Running image detection..."):
            result_image = run_inference(model, input_image, is_video=False)

        st.subheader("Detected Output:")
        st.image(result_image, use_column_width=True)

        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Result Image",
            data=byte_im,
            file_name="result_image.png",
            mime="image/png"
        )

    # VIDEO
    elif "video" in file_type:
        st.subheader("Input Video:")
        st.video(uploaded_file)

        with st.spinner("Running video detection..."):
            output_path = run_inference(model, uploaded_file, is_video=True)

        st.subheader("Detected Output Video:")
        st.video(output_path)

        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.download_button(
                label="📥 Download Result Video",
                data=video_bytes,
                file_name="result_video.mp4",
                mime="video/mp4"
            )
