import re
import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def detect_emotions(video_path, output_path, images_path, qtd_frames, model_name):
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

    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
    
        # Ler um frame do vídeo
        ret, frame = cap.read()
        
        original_frame = frame.copy()
        
        result = None

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break
        
        if frame_counter == 0:
            
            # Analisar o frame para detectar faces e expressões
            #result = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            result = DeepFace.analyze(original_frame, actions=['age','gender','emotion'], enforce_detection=False, detector_backend='mtcnn', align=True)
            
            # se nenhuma faace for detectada, rotaciona a imagem e tenta novamente
            if result == None:
                rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                result = DeepFace.analyze(rotated_frame, actions=['age','gender','emotion'], enforce_detection=False, detector_backend='mtcnn', align=True)

            # Iterar sobre cada face detectada
            for face in result:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                
                # recortar a face do frame
                face_img = original_frame[y:y+h, x:x+w]
                
                # Mostrar a imagem da face recortada
                cv2.imshow("Face", face_img)
                
                # reconhecimento da pessoa da imagem no banco de imagens utilizando do deepface
                result = DeepFace.find(face_img, db_path = images_path, enforce_detection=False, model_name = model_name)
                
                # Se não houver nenhuma face detectada, salvar a imagem na pasta de banco de imagens    
                if result[0].empty:
                    # recortar a face do frame
                    offset = 80                    
                    new_Y = y -offset if y > offset else y
                    new_H = y+h + offset
                    new_x = x -offset if x > offset else x
                    new_w = x+w +offset
                    
                    face_img_to_save = original_frame[new_Y:new_H,new_x:new_w]
                    
                    # Mostrar a imagem da face recortada
                    cv2.imshow("Face", face_img_to_save)
                
                    cv2.imwrite(f"{images_path}/pessoa_"+str(detected_face_counter)+".jpg", face_img_to_save)
                    detected_face_counter += 1
                    person = f"pessoa_"+str(detected_face_counter)
                else:
                    person = result[0].identity
                    # Extrair o nome da imagem
                    _str =  str(person.values[0])
                    match = re.search(r'images2\\(.*?).jpg', _str)
                    person = match.group(1)
                
                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
                # Obter a emoção dominante
                name = person
                age = face['age']
                dominant_emotion = face['dominant_emotion']
                gender = face['dominant_gender']
                #race = face['dominant_race']
                
                
                # Escrever a emoção dominante, idade, gênero e raça acima da face
                #text = f"{dominant_emotion}, {age}, {gender}, {race}"
                text = f"{name}, {age} anos, {gender}, {dominant_emotion},"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                # Escrever a emoção dominante acima da face
                #cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                #cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                frame_counter += 1
                
        else:
            
            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Escrever a emoção dominante, idade, gênero e raça acima da face
            #text = f"{dominant_emotion}, {age}, {gender}, {race}"
            text = f"{name}, {age} anos, {gender}, {dominant_emotion},"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Escrever o frame processado no vídeo de saída
        out.write(frame)
        
        frame_counter += 1
        
        if frame_counter >= qtd_frames:
                frame_counter = 0


    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'videos/original_video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'videos/video_expression_detect_retina_face.mp4')  # Nome do vídeo de saída
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

