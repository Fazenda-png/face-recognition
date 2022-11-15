import cv2 as cv

faceCascade = cv.CascadeClassifier("models/haarcascade_frontalface_alt.xml")

#Captura de video
camera = cv.VideoCapture(0 , cv.CAP_DSHOW)
#Redimencionando o video
camera.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 1024)

print('Press (S) to Save Image...')


while True:

    #Lendo a camera
    _, frame = camera.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    objetos = faceCascade.detectMultiScale(gray, 1.2, 5 )

    for (x, y, w, h) in objetos:
        cv.rectangle(frame, (x - 100, y - 100), (x+w + 100, y+h + 100), (0, 0, 255), 2)
        crop = frame[y- 100 :y+h + 100, x - 100:x+w + 100]

    #Lendo a camera
    cv.imshow("Camera",frame)

   # Aperte "ESC" para sair
    key = cv.waitKey(60)
    if key == 27:
        break
    elif key == ord("s"):
        # Salvar a imagem cortada na pasta data_base
        print("Coloque o nome completo")
        print("ATENÇÃO:\n" "1) Não coloque acentos no seu nome \n" "2) coloque espaço entre as palavras")
        name = input("Qual o seu nome? ")
        cv.imwrite("./data_base/" + name + ".jpg", crop)
        print("não deseja adcionar mais? aperte ESC")

cv.destroyAllWindows()