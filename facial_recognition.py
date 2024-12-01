# pip install deepface keras tensorflow opencv-python-headless tqdm numpy matplotlib dlib mtcnn keras_vggface keras-models keras-layers
import os
import re
import time

import cv2
from deepface import DeepFace
from tqdm import tqdm


def detect_emotions(video_path, output_path, images_path, qtd_frames, model_name):
    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    detected_face_counter = 1
    frame_counter = 0
    total_numer_dif_faces = 0

    # Processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        original_frame = frame.copy()

        if frame_counter == 0:
            result = DeepFace.analyze(
                original_frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='mtcnn',
                align=True
            )

            # Tentar com rotação se nenhuma face for detectada
            if result is None:
                rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                result = DeepFace.analyze(
                    rotated_frame,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False,
                    detector_backend='mtcnn',
                    align=True
                )

            if result is not None:
                for face in result:
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    face_img = original_frame[y:y + h, x:x + w]

                    result = DeepFace.find(
                        face_img, db_path=images_path,
                        enforce_detection=False, model_name=model_name
                    )

                    emotion = face['dominant_emotion']
                    emotion_color = {
                        "happy": (0, 255, 0),
                        "sad": (255, 0, 0),
                        "angry": (0, 0, 255),
                        "surprise": (255, 255, 0),
                        "neutral": (255, 255, 255)
                    }.get(emotion, (36, 255, 12))  # Default color

                    if result[0].empty:
                        offset = 80
                        face_img_to_save = original_frame[
                                           max(0, y - offset):y + h + offset,
                                           max(0, x - offset):x + w + offset
                                           ]
                        cv2.imwrite(f"{images_path}/pessoa_{detected_face_counter}.jpg", face_img_to_save)
                        person = f"pessoa_{detected_face_counter}"
                        detected_face_counter += 1
                    else:
                        person = re.search(r'images2\\(.*?).jpg', str(result[0].identity.values[0])).group(1)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)
                    text = f"{person}, {face['age']} anos, {face['dominant_gender']}, {face['dominant_emotion']}"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

                    # display the frame
                    # cv2.imshow("Frame", frame)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)
        frame_counter = (frame_counter + 1) % qtd_frames

    # Liberar recursos
    # Printa numero total de frames analisados
    print(f"Total de frames analisados: {total_frames}")
    print(f"Total de faces detectadas: {detected_face_counter - 1}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    # Inicia a contagem do tempo total de processamento
    start_time = time.time()
    print("Iniciando a detecção de emoções...")
    # Configurar caminhos e parâmetros
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'videos/original_video.mp4')
    output_video_path = os.path.join(script_dir, 'videos/output_face_detection_video.mp4')
    images_path = os.path.join(script_dir, 'images2')
    qtd_frames = 10

    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
        "GhostFaceNet",
    ]

    # Detetar emoções no vídeo
    detect_emotions(input_video_path, output_video_path, images_path, qtd_frames, models[0])

    # Printa a hora de término
    print("Detecção de emoções finalizada.")
    end_time = time.time()

    # Mostra o tempo total de processamento
    print(f"Tempo total de processamento: {end_time - start_time:.2f} segundos.")


if __name__ == '__main__':
    main()
