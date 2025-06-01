import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import os
import tempfile
from model import get_model  # Import your model definition

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model():
    model = get_model(device)
    model.load_state_dict(torch.load("deepfake_model_made_using_jupyternb.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Preprocessing transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to split video into frames
def extract_frames(video_path, frame_limit=30, resize_dim=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= frame_limit:  # Stop after `frame_limit` frames
            break
        frame_count += 1

        # Resize frame to desired dimensions
        frame = cv2.resize(frame, resize_dim)

        # Convert frame from BGR to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()
    return frames

# Function to process frames and predict
def process_frames(frames):
    predictions = []

    for frame in frames:
        # Apply transformations
        frame_tensor = transform(frame).unsqueeze(0).to(device)

        # Predict using the model
        with torch.no_grad():
            output = model(frame_tensor)
            pred = torch.sigmoid(output).item()
            predictions.append(pred)

    # Calculate average prediction
    avg_prediction = sum(predictions) / len(predictions) if predictions else 0.0
    return avg_prediction

# Streamlit UI
st.title("Deepfake Detection App")
st.write("Upload a video, and the model will determine whether it is a deepfake.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    temp_video_path = temp_file.name

    try:
        st.video(temp_video_path)

        st.write("Extracting frames from the video...")
        frames = extract_frames(temp_video_path)

        st.write(f"Processing {len(frames)} frames...")
        avg_prediction = process_frames(frames)

        # Display the result
        if avg_prediction > 0.5:
            st.error(f"The video is likely a deepfake. Confidence: {avg_prediction:.2f}")
        else:
            st.success(f"The video is likely authentic. Confidence: {1 - avg_prediction:.2f}")

    finally:
        # Clean up
        # os.remove(temp_video_path)
        print("done")
