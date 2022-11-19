#Imports
import face_recognition
import cv2
import numpy as np
import pytesseract
import os
import glob
from datetime import datetime

#Iniciando Listas
faces_encodings = []
faces_names = []
list_of_files = []

#Diretorio atual
cur_direc = os.getcwd()

#Diretorio com as imagens para detecção
path = os.path.join(cur_direc, 'data_base/')

#Loop para pegar os aquivos e armazenar na variavel list_of_files
list_of_files = [f for f in glob.glob(path+'*.jpg')]

print(list_of_files)

#Numero de arquivos na lista
number_files = len(list_of_files)

#Copiado a lista
names = list_of_files.copy()

#Config data e hora
data_hora = datetime.now()

#Loop com o total de imagens na pasta
for i in range(number_files):

    #Carrega o arquivo e retorna uma matriz
    presets = face_recognition.load_image_file(list_of_files[i])

    #Retorna uma lista com recursos que representam locais do rosto
    encoding = face_recognition.face_encodings(presets)[0]
    faces_encodings.append(encoding)

    #Pegar o nome do arquivo e adicionar na lista
    nome = names[i].replace(cur_direc+"\\data_base\\", "")
    nome = nome.replace(".jpg", "")
    names[i] = nome
    faces_names.append(names[i])

#Iniciando Listas
face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

#Captura de video
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Redimencionando o video
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

#Cores para detecção
lowerBlue = np.array([9, 97, 103])
upperBlue = np.array([255, 255, 255])

while True:

    #Lendo a camera
    _, frame = video_capture.read()

    #Mudança de cores
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Estabelecendo um range do frame com (COLOR_BGR2HSV) e das cores setadas
    mask = cv2.inRange(hsv, lowerBlue, upperBlue)
    
    #Comparação bit a bit
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #Mudança de cores
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #Binarização
    _, borda = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    #Localizando o contorno
    contornos, _ = cv2.findContours(
        borda, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #Redimencionamento do frame
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:

            #Comparação entre o atual com o da imagem
            matches = face_recognition.compare_faces(
                faces_encodings, face_encoding)
            name = "Desconhecido"

            #Um matriz com a distância para cada face na mesma ordem que a matriz 'faces'
            face_distances = face_recognition.face_distance(
                faces_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)

            for contorno in contornos:

                area = cv2.contourArea(contorno)
                #Condição para marcar areas com mais de 400px
                if area > 400:
                    (x, y, w, h) = cv2.boundingRect(contorno)
                    #Função que desenha um retangulo nas coordenadas x,y,w,h
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    #Função escreve na imagem
                    cv2.putText(
                        frame,
                        str("Cracha"),
                        (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 1
                    )
                    nome_arquivo = name + '_' + data_hora.strftime("%d_%m_%Y %H_%M_%S")
                    cv2.imwrite("./log_ponto/" + nome_arquivo + ".jpg", frame)


    cv2.imshow('Ponto eletronico', frame)
    # Aperte "q" para sair
    k = cv2.waitKey(60)
    if k == 27:
        break

cv2.destroyAllWindows()
video_capture.release()