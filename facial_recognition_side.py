# Required installations:
# pip install opencv-python-headless deepface tf-keras tqdm deepface
import os
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace


def detect_emotions_and_side_faces(video_path, output_path):
    """
    Perform emotion detection and side-face detection on a video,
    save the output video, and return the total frames analyzed.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.

    Returns:
        int: Total number of frames analyzed.
    """
    # Load cascades for side-face detection
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    # Open the input video
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    # Process each frame of the video
    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Skip processing for some frames to save computation
        if frame_idx % 2 != 0:
            out.write(frame)  # Write unprocessed frames to the output
            continue

        try:
            # Analyze the frame for face and emotion detection
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Convert frame to grayscale for side-face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect side faces
            side_faces = profile_face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            # Process each face detected by DeepFace
            for face in results:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                # Ensure bounding box dimensions are within valid range
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x - 1)
                h = min(h, height - y - 1)

                # Get the dominant emotion and corresponding color
                dominant_emotion = face['dominant_emotion']
                color_map = {
                    "happy": (0, 255, 0),  # Green
                    "sad": (255, 0, 0),  # Blue
                    "angry": (0, 0, 255),  # Red
                    "surprise": (255, 255, 0),  # Yellow
                    "neutral": (255, 255, 255)  # White
                }
                emotion_color = color_map.get(dominant_emotion, (36, 255, 12))  # Default color

                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

            # Process side faces
            for (x, y, w, h) in side_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for side faces
                cv2.putText(frame, "Side Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Write the processed frame to the output video
            out.write(frame)

        except Exception as e:
            print(f"Warning: An error occurred while processing a frame: {e}")
            out.write(frame)  # Write the unprocessed frame if analysis fails

        frame_count += 1

    # Release video objects and cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return frame_count