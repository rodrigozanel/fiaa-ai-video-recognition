import os

import activity_detection as ad
import activity_detection_actions as adw
import facial_recognition as fr
import video_transcription as vt


def main():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths for input and output files
    original_video_path = os.path.join(
        script_dir, 'resources/Unlocking Facial Recognition_ Diverse Activities Analysis.mp4'
    )

    # original_video_path = os.path.join(
    #    script_dir, 'resources/WIN_20241126_19_05_14_Pro.mp4'
    # )

    output_video_emotions = os.path.join(script_dir, 'resources/output_video.mp4')
    output_video_pose = os.path.join(script_dir, 'resources/output_video_pose.mp4')
    output_video_pose_count = os.path.join(script_dir, 'resources/output_video_pose_count.mp4')
    output_audio_path = os.path.join(script_dir, 'resources/audio1.wav')
    text_output_path = os.path.join(script_dir, 'resources/transcription_audio.txt')

    # Step 1: Detect emotions in the video
    #frame_count = fr.detect_emotions_with_rotations(original_video_path, output_video_emotions)
    #print(f"Total frames analyzed: {frame_count}")

    # Step 1.2: Detect side faces in the video
    # frame_count = frs.detect_emotions_and_side_faces(original_video_path, output_video_emotions)

    # Step 2: Detect poses in the video
    ad.detect_pose(output_video_emotions, output_video_pose)

    # Step 3: Detect and count actions (e.g., waving) in the video
    adw.detect_pose_and_count_actions(output_video_pose, output_video_pose_count)

    # Step 4: Extract audio from the original video
    vt.extract_audio_from_video(original_video_path, output_audio_path)

    # Step 5: Transcribe audio to text
    vt.transcribe_audio_to_text(output_audio_path, text_output_path)


if __name__ == '__main__':
    main()
