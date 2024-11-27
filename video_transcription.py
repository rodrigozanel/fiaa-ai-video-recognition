# Required installations:
# pip install moviepy SpeechRecognition pydub
import moviepy as mp
import speech_recognition as sr


def extract_audio_from_video(video_path, audio_path):
    """
    Extract the audio from a video file and save it as a separate audio file.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file.
    """
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def transcribe_audio_to_text(audio_path, text_output_path):
    """
    Transcribe speech from an audio file to text and save the text to a file.

    Args:
        audio_path (str): Path to the audio file to be transcribed.
        text_output_path (str): Path to save the transcription text file.
    """
    recognizer = sr.Recognizer()

    try:
        # Load the audio file
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)  # Read the entire audio file

            # Use Google Speech Recognition to transcribe the audio
            text = recognizer.recognize_google(audio, language="en-US")
            print("Transcription: " + text)

            # Save the transcription to a text file
            with open(text_output_path, 'w', encoding='utf-8') as file:
                file.write(text)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Error requesting results from Google Speech Recognition service; {e}")