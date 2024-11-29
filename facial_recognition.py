# Required installations:
# pip install opencv-python-headless tqdm deepface
import os
import uuid

import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
from tqdm import tqdm


def detect_emotions_with_rotations(video_path, output_path, library_path="library"):
    """
    Detect emotions and side faces in a video using frame rotation.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video.

    Returns:
        int: Total number of frames analyzed.
    """
    # Load Haar Cascade for side-face detection
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if profile_face_cascade.empty():
        raise ValueError("Haar Cascade for profile faces could not be loaded. Check the path.")

    # Open the video file
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
    print(f"Total frames: {total_frames}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        #if frame_idx % 2 != 0:  # Skip every other frame to reduce computation
        #    out.write(frame)
        #    continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Redimension the frame to 1/4 of the size
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])  # Convert BGR to RGB

        face_locations = face_recognition.face_locations(rgb_small_frame)  # Localizar faces no frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Obter codificações faciais

        face_names = []  # Lista para armazenar os nomes das faces detectadas
        for face_encoding in face_encodings:
            name = search_faces_in_folder('library', face_encoding)
            if (name):
                face_names.append(name)
            else:
                random_filename = str(uuid.uuid4()) + ".jpg"
                face_names.append(random_filename)
                # Save the detected face to the library
                full_path = os.path.join(library_path, random_filename)
                os.makedirs(library_path, exist_ok=True)
                cv2.imwrite(full_path, small_frame)
                print(f"New face detected and saved to the library: {random_filename}")

        try:
            # DeepFace analysis for the original frame
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Detect side faces using Haar cascade
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            side_faces = profile_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5,
                                                               minSize=(50, 50))

            # Draw bounding boxes for side faces
            for (x, y, w, h) in side_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Side Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Process each face detected by DeepFace
            for face in results:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                emotion = face['dominant_emotion']
                emotion_color = {
                    "happy": (0, 255, 0),
                    "sad": (255, 0, 0),
                    "angry": (0, 0, 255),
                    "surprise": (255, 255, 0),
                    "neutral": (255, 255, 255)
                }.get(emotion, (36, 255, 12))  # Default color

                cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
        except Exception as e:
            print(f"Warning: An error occurred on frame {frame_idx}: {e}")

        # Write the processed frame to the output
        out.write(frame)
        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return frame_count


def search_faces_in_folder(folder, face_encoding):
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encoding = face_encodings[0]
                result = face_recognition.compare_faces([face_encoding], known_face_encoding)
                if result[0]:
                    return filename
    return None
