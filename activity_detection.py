# Required installations:
# pip install opencv-python mediapipe tqdm
import cv2
import mediapipe as mp
from tqdm import tqdm


def detect_pose(video_path, output_path):
    """
    Detect human poses in a video, annotate the frames with pose landmarks, and save the output video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output annotated video.
    """
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open the input video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process the video frame by frame with a progress bar
    for _ in tqdm(range(total_frames), desc="Processing video"):
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if the frame couldn't be read (end of video)
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect pose landmarks
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Write the annotated frame to the output video
        out.write(frame)

        # Optional: Display the frame (disabled for non-interactive environments)
        # cv2.imshow('Pose Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()