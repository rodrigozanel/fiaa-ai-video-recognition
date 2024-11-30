import cv2
import mediapipe as mp
from tqdm import tqdm

def detect_pose_and_count_actions(video_path, output_path):
    """
    Detecta poses e conta ações (movimentos de braço, acenos, apertos de mão, acenos de cabeça) em um vídeo,
    salvando o vídeo de saída com anotações.

    Args:
        video_path (str): Caminho para o arquivo de vídeo de entrada.
        output_path (str): Caminho para salvar o vídeo de saída anotado.
    """
    # Inicializa os utilitários do MediaPipe para pose e desenho
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Abre o arquivo de vídeo de entrada
    cap = cv2.VideoCapture(video_path)

    # Verifica se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        print("Erro: Não foi possível abrir o arquivo de vídeo.")
        return

    # Obtém as propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define o codec e cria o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Inicializa contadores e flags
    arm_up = False
    arm_movements_count = 0
    wave = False
    wave_count = 0
    handshake = False
    handshake_count = 0
    nod = False
    nod_count = 0

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

    # Processa cada quadro do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break  # Fim do vídeo

        # Converte o quadro para RGB para processamento com MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa o quadro para detectar os marcos da pose
        results = pose.process(rgb_frame)

        # Mostra o frame
        cv2.imshow("Frame", frame)

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
            cv2.putText(frame, f'Acenos: {wave_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Apertos de Mao: {handshake_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Acenos de Cabeca: {nod_count}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)

        # Escreve o quadro processado no vídeo de saída
        out.write(frame)

    # Libera os recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

