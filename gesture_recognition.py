import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# Inicializa contadores e flags
arm_up = False
arm_movements_count = 0
wave = False
wave_count = 0
handshake = False
handshake_count = 0
nod = False
nod_count = 0
anomalos_count = 0


# Função auxiliar: Verifica se um braço está levantado
def is_arm_up(landmarks):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y

    return left_arm_up or right_arm_up


# Função auxiliar: Detecta gesto de aceno
def is_wave(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    return (left_wrist.y < left_elbow.y) or (right_wrist.y < right_elbow.y)


# Função auxiliar: Detecta gesto de aperto de mão
def is_handshake(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return (left_wrist.x < left_shoulder.x) and (right_wrist.x > right_shoulder.x)


# Função auxiliar: Detecta gesto de aceno de cabeça
def is_nod(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return nose.y < ((left_shoulder.y + right_shoulder.y) / 2)


# from mediapipe.framework.formats import image_frame
# from mediapipe.tasks.python.vision import Image

def detect(recognizer):
    # Configuração do vídeo
    hora_inicio = time.time()
    video_path = 'videos/video_expression_detected.mp4'
    output_path = 'videos/video_expression_and_pose_detected.mp4'
    cap = cv2.VideoCapture(video_path)
    latest_gesture = None

    if not cap.isOpened():
        print("Erro: Não foi possível abrir o arquivo de vídeo.")
        exit()

    # Propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configuração do VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Ferramenta para desenhar
    drawing_utils = mp.solutions.drawing_utils

    # Inicializa o último timestamp
    last_timestamp_ms = 0
    gestures = []
    arm_movements_count = 0
    arm_up = False
    wave_count = 0
    wave = False
    handshake_count = 0
    handshake = False
    nod_count = 0
    nod = False
    prev_positions = None

    # Processa cada quadro
    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # A cada 120 frames, limpa o último gesto detectado
        if frame_idx % 120 == 0:
            latest_gesture = None

        # Se ha um gesto detectado, exibe-o no quadro (Em Laranja escuro),  mas somente por 120 frames
        if latest_gesture:
            cv2.putText(frame, f"Ultimo Gesto Detectado: {latest_gesture}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 140, 255), 2, cv2.LINE_AA)

        original_frame = frame.copy()

        # Calcula o timestamp em milissegundos com base no índice do frame e no FPS
        timestamp_ms = int((frame_idx / fps) * 1000)
        if (timestamp_ms == 0):
            continue

        # Converte o quadro para RGB para processamento com MediaPipe
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Mostra o video
        # cv2.imshow('Video', frame)

        # Reconhecimento de gestos
        recognition_result = recognizer.recognize_for_video(rgb_frame, timestamp_ms)

        # Desenha landmarks e gestos
        # if recognition_result.hand_landmarks:
        # for hand_landmark in recognition_result.hand_landmarks:
        # drawing_utils.draw_landmarks(frame, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS)

        if recognition_result.gestures:
            for gesture in recognition_result.gestures:
                for i in range(len(gesture)):
                    top_gesture = gesture[i].category_name
                    if (top_gesture == "None"):
                        continue
                    latest_gesture = top_gesture
                    print(f"Gesto: {gesture[i].category_name} - Probabilidade: {gesture[i].score}")
                    # Se o gesto ja foi detectado, altera a flag para True
                    # adiciona o gesto à lista de gestos incrementando a quantidade de vezes que ele foi detectado
                    gestures.append(gesture[i].category_name)
                    cv2.putText(frame, f"Gesto: {top_gesture}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255),
                                2, cv2.LINE_AA)

        # Agora manualmente
        # Processa o quadro para detectar os marcos da pose
        results = pose.process(original_frame)

        # Mostra o frame
        # cv2.imshow("Frame", frame)

        if results.pose_landmarks:
            # Desenha os marcos da pose no quadro
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Conta movimentos de braço
            if is_arm_up(results.pose_landmarks.landmark):
                if not arm_up:
                    arm_up = True
                    arm_movements_count += 1
            else:
                arm_up = False

            # Conta gestos de aceno
            if is_wave(results.pose_landmarks.landmark):
                if not wave:
                    wave = True
                    wave_count += 1
            else:
                wave = False

            # Conta gestos de aperto de mão
            if is_handshake(results.pose_landmarks.landmark):
                if not handshake:
                    handshake = True
                    handshake_count += 1
            else:
                handshake = False

            # Conta gestos de aceno de cabeça
            if is_nod(results.pose_landmarks.landmark):
                if not nod:
                    nod = True
                    nod_count += 1
            else:
                nod = False

            # Exibe os contadores no quadro
            cv2.putText(frame, f'Bracos levantados: {arm_movements_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            # print(f'Bracos levantados: {arm_movements_count}')
            cv2.putText(frame, f'Movimentos de Maos: {wave_count + handshake_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            # print(f'Acenos: {wave_count}')
            # print(f'Apertos de Mao: {handshake_count}')
            cv2.putText(frame, f'Movimentos de Cabeca: {nod_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            # print(f'Acenos de Cabeca: {nod_count}')

        # Escreve o quadro no vídeo de saída
        out.write(frame)

    # Exibe os gestos detectados removendo duplicatas e exibindo a quantidade de vezes que cada gesto foi detectado
    print("Gestos detectados:")
    for gesture in set(gestures):
        print(f"{gesture}: {gestures.count(gesture)}")

    hora_fim = time.time()
    # anomalos_count 'e um distinct count de gestos
    anomalos_count = len(set(gestures))
    print(f"Identificao de Gestos e Movimentos - Tempo total de processamento: {hora_fim - hora_inicio:.2f} segundos")
    print(f"Identificao de Gestos e Movimentos - Total de frames analisados: {total_frames}")
    print(f"Identificao de Gestos e Movimentos - Total de gestos detectados: {len(gestures)}")
    print(f"Identificao de Gestos e Movimentos - Hora de inicio: {hora_inicio}")
    print(f"Identificao de Gestos e Movimentos - Hora de fim: {hora_fim}")
    print(f"Identificao de Gestos e Movimentos - Tempo de execução: {hora_fim - hora_inicio}")
    print(f"Identificao de Gestos e Movimentos - Bracos levantados: {arm_movements_count}")
    print(f"Identificao de Gestos e Movimentos - Movimentos de Maos: {wave_count + handshake_count}")
    print(f"Identificao de Gestos e Movimentos - Movimentos de Cabeca: {nod_count}")
    print(f"Identificao de Gestos e Movimentos - Movimentos Anomalos: {anomalos_count}")

    with open('processing_summary.txt', 'w') as f:
        f.write(f"Identificao de Gestos e Movimentos - Total de frames analisados: {total_frames}\n")
        f.write(f"Identificao de Gestos e Movimentos - Total de gestos detectados: {len(gestures)}\n")
        f.write(f"Identificao de Gestos e Movimentos - Hora de inicio: {hora_inicio}\n")
        f.write(f"Identificao de Gestos e Movimentos - Hora de fim: {hora_fim}\n")
        f.write(f"Identificao de Gestos e Movimentos - Tempo de execução: {hora_fim - hora_inicio}\n")
        f.write(f"Identificao de Gestos e Movimentos - Bracos levantados: {arm_movements_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos de Maos: {wave_count + handshake_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos de Cabeca: {nod_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos Anomalos: {anomalos_count}\n")

    # Libera recursos
    cap.release()
    out.release()
    print("Processamento concluído.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Inicializa os utilitários do MediaPipe para pose e desenho
    model_asset_path = os.path.join(script_dir, 'models/gesture_recognizer.task')
    # Inicialização do modelo GestureRecognizer
    VisionRunningMode = vision.RunningMode.VIDEO
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.GestureRecognizerOptions(base_options=base_options, running_mode=VisionRunningMode, num_hands=2,
                                              min_tracking_confidence=0.5, min_hand_presence_confidence=0.5,
                                              min_hand_detection_confidence=0.5)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    detect(recognizer)
