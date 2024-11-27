# Required installations:
# pip install opencv-python-headless deepface tf-keras tqdm deepface
import os
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace


def load_images_from_folder(folder):
    """
    Load images from a specified folder and extract face encodings and names.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        tuple: A tuple containing:
            - known_face_encodings (list): List of face encodings.
            - known_face_names (list): List of face names.
    """
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                face_encoding = face_encodings[0]
                name = os.path.splitext(filename)[0][:-1]  # Remove numeric suffix and extension
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    return known_face_encodings, known_face_names


def detect_emotions(video_path, output_path):
    """
    Perform emotion detection on a video, saving the output video and returning the total frames analyzed.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.

    Returns:
        int: Total number of frames analyzed.
    """
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
        if frame_idx % 30 != 0:
            out.write(frame)  # Write unprocessed frames to the output
            continue

        try:
            # Analyze the frame for face and emotion detection
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Process each face detected in the frame
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