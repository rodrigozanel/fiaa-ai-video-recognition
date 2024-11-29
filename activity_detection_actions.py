import cv2
import mediapipe as mp
from tqdm import tqdm


def detect_pose_and_count_actions(video_path, output_path):
    """
    Detect poses and count actions (arm movements, waves, handshakes, nods) in a video,
    and save the output video with annotations.

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

    # Initialize counters and flags
    arm_up = False
    arm_movements_count = 0

    wave = False
    wave_count = 0

    handshake = False
    handshake_count = 0

    nod = False
    nod_count = 0

    # Helper function: Check if an arm is raised
    def is_arm_up(landmarks):
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        left_arm_up = left_elbow.y < left_eye.y
        right_arm_up = right_elbow.y < right_eye.y

        return left_arm_up or right_arm_up

    # Placeholder function: Detect a wave gesture
    def is_wave(landmarks):
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y and \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y:
            return True
        return is_arm_up(landmarks)

    # Placeholder function: Detect a handshake
    def is_handshake(landmarks):
        if landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x and \
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x:
            return True
        return False

    # Placeholder function: Detect a nod gesture
    def is_nod(landmarks):
        if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y and \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y:
            return True
        return False


    def arms_down(landmarks):
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        left_arm_down = left_elbow.y > left_eye.y
        right_arm_down = right_elbow.y > right_eye.y

        return left_arm_down and right_arm_down

    # Process each frame of the video with a progress bar
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect pose landmarks
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Count arm movements
            if is_arm_up(results.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
            else:
                arm_up = False

            # Count wave gestures
            if is_wave(results.pose_landmarks.landmark):
                if not wave:
                    wave = True
                    wave_count += 1
            else:
                wave = False

            # Count handshake gestures
            if is_handshake(results.pose_landmarks.landmark):
                if not handshake:
                    handshake = True
                    handshake_count += 1
            else:
                handshake = False

            # Count nod gestures
            if is_nod(results.pose_landmarks.landmark):
                if not nod:
                    nod = True
                    nod_count += 1
            else:
                nod = False

            # Display counters on the frame
            cv2.putText(frame, f'Arm Movements: {arm_movements_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Waves: {wave_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Handshakes: {handshake_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Nods: {nod_count}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the processed frame to the output video
        out.write(frame)

        # Optional: Display the frame (disabled for non-interactive environments)
        # cv2.imshow('Pose Detection', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
