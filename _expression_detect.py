import os
import re
import time

import cv2
from deepface import DeepFace
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


def detect_emotions(video_path, output_path, images_path, qtd_frames, model_name):
    hora_inicio = time.time()
    # Inicialização do modelo GestureRecognizer
    VisionRunningMode = vision.RunningMode.VIDEO
    base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_face_counter = 1
    age = 0
    text = ""
    frame = None
    frame_counter = 0
    gestures = []

    # Loop para processar cada frame do vídeo
    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):

        # Ler um frame do vídeo
        ret, frame = cap.read()

        original_frame = frame.copy()

        result = None

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        if frame_counter == 0:

            # Analisar o frame para detectar faces e expressões
            # result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            result = DeepFace.analyze(original_frame, actions=['age', 'gender', 'emotion'], enforce_detection=False,
                                      detector_backend='mtcnn', align=True)

            # se nenhuma faace for detectada, rotaciona a imagem e tenta novamente
            if result == None:
                rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                result = DeepFace.analyze(rotated_frame, actions=['age', 'gender', 'emotion'], enforce_detection=False,
                                          detector_backend='mtcnn', align=True)

            # Iterar sobre cada face detectada
            for face in result:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                # recortar a face do frame
                face_img = original_frame[y:y + h, x:x + w]

                # Mostrar a imagem da face recortada
                # cv2.imshow("Face", face_img)

                # reconhecimento da pessoa da imagem no banco de imagens utilizando do deepface
                result = DeepFace.find(face_img, db_path=images_path, enforce_detection=False, model_name=model_name)

                emotion = face['dominant_emotion']
                emotion_color = {
                    "happy": (0, 255, 0),
                    "sad": (255, 0, 0),
                    "angry": (0, 0, 255),
                    "surprise": (255, 255, 0),
                    "neutral": (255, 255, 255)
                }.get(emotion, (36, 255, 12))  # Default color

                # Se não houver nenhuma face detectada, salvar a imagem na pasta de banco de imagens    
                if result[0].empty:
                    # recortar a face do frame
                    offset = 80
                    new_Y = y - offset if y > offset else y
                    new_H = y + h + offset
                    new_x = x - offset if x > offset else x
                    new_w = x + w + offset

                    face_img_to_save = original_frame[new_Y:new_H, new_x:new_w]

                    # Mostrar a imagem da face recortada
                    # cv2.imshow("Face", face_img_to_save)

                    cv2.imwrite(f"{images_path}/pessoa_" + str(detected_face_counter) + ".jpg", face_img_to_save)
                    detected_face_counter += 1
                    person = f"pessoa_" + str(detected_face_counter)
                else:
                    person = result[0].identity
                    # Extrair o nome da imagem
                    _str = str(person.values[0])
                    match = re.search(r'images2\\(.*?).jpg', _str)
                    person = match.group(1)

                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)

                # Obter a emoção dominante
                name = person
                age = face['age']
                dominant_emotion = face['dominant_emotion']
                gender = face['dominant_gender']
                # race = face['dominant_race']

                # Escrever a emoção dominante, idade, gênero e raça acima da face
                # text = f"{dominant_emotion}, {age}, {gender}, {race}"
                text = f"{name}, {age} anos, {gender}, {dominant_emotion},"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
                # Escrever a emoção dominante acima da face
                # cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Mostra o video
                # cv2.imshow('Video', frame)

                frame_counter += 1

        else:

            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Escrever a emoção dominante, idade, gênero e raça acima da face
            # text = f"{dominant_emotion}, {age}, {gender}, {race}"
            text = f"{name}, {age} anos, {gender}, {dominant_emotion},"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Reconhecimento de gestos
        # Calcula o timestamp em milissegundos com base no índice do frame e no FPS
        # timestamp_ms = int((frame_idx / fps) * 1000)
        # if (timestamp_ms == 0):
        #    continue

        # Converte o quadro para RGB para processamento com MediaPipe
        # rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=original_frame)
        # recognition_result = recognizer.recognize_for_video(rgb_frame, timestamp_ms)

        # Desenha landmarks e gestos
        # if recognition_result.hand_landmarks:
        # for hand_landmark in recognition_result.hand_landmarks:
        # drawing_utils.draw_landmarks(frame, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS)

        # if recognition_result.gestures:
        #    for gesture in recognition_result.gestures:
        #        for i in range(len(gesture)):
        #            top_gesture = gesture[i].category_name
        #            print(f"Gesto: {gesture[i].category_name} - Probabilidade: {gesture[i].score}")
        #            if (top_gesture == "None"):
        #                continue
        #            # adiciona o gesto à lista de gestos incrementando a quantidade de vezes que ele foi detectado
        #            gestures.append(gesture[i].category_name)
        #            cv2.putText(frame, f"Gesto: {top_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                        (0, 255, 0), 2)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        frame_counter += 1

        if frame_counter >= qtd_frames:
            frame_counter = 0

    hora_fim = time.time()
    # Liberar a captura de vídeo e fechar todas as janelas
    # Exibe o total de frames analisados
    print(f"Total de frames analisados: {total_frames}")
    print(f"Total de faces detectadas: {detected_face_counter - 1}")
    print(f"Hora de inicio: {hora_inicio}")
    print(f"Hora de fim: {hora_fim}")
    print(f"Tempo de execução: {hora_fim - hora_inicio}")

    # Salva os dados de frames analisados e faces detectadas em um arquivo (append)
    with open('processing_summary.txt', 'w') as f:
        f.write(f"Detecao de Faces - Total de frames analisados: {total_frames}\n")
        f.write(f"Detecao de Faces - Total de faces detectadas: {detected_face_counter - 1}\n")
        f.write(f"Detecao de Faces - Hora de inicio: {hora_inicio}\n")
        f.write(f"Detecao de Faces - Hora de fim: {hora_fim}\n")
        f.write(f"Detecao de Faces - Tempo de execução: {hora_fim - hora_inicio}\n")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir,
                                'videos/original_video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'videos/video_expression_detected.mp4')  # Nome do vídeo de saída
images_path = os.path.join(script_dir, 'images2')  # Nome do vídeo de saída
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

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'fastmtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

alignment_modes = [True, False]

# Chamar a função para detectar emoções no vídeo e salvar o vídeo processado
detect_emotions(input_video_path, output_video_path, images_path, qtd_frames, models[0])
