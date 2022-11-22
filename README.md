# Ponto Automatizado

Trabalho realizado por: João Pedro Fazenda Dos Santos (1117210), Eduardo Danelli (1121477)

O sistema é dividido em duas partes:
cadastro_func_foto.py = seria para realizar o cadastro da pessoa na base de dados. Consiste em tirar uma foto
e colocar o nome da pessoa.
Nesta etapa utilizamos um CascadeClassifier com modelos de um rosto de frente para a detecção, após isso 
implementamos a aplicação de borda no rosto e assim realizamos o corte utilizando o crop para salvar a imagem
na pasta data_base.
Resultado:
recorte da imagem
![image](https://user-images.githubusercontent.com/57419268/203354273-a0a101e8-830e-4c56-8a28-9639b20c3250.png)


ponto.py = basicamente a camera ira detectar o rosto e fazer o reconhecimento dele e verificar se o funcionario
está usando o crachá fazendo uma seleção de cor e assim que ele detectar a cor ele ira logar a imagem da camera
na pasta log_ponto com uma imagem da pessoa, o nome dela e o horario em que ela passou.
reconhecimento facial + seleção da cor do crachá
![image](https://user-images.githubusercontent.com/57419268/203354537-5038a462-5b26-4e31-ba99-5bdde6a5b3dc.png)

Libs: 
  OpenCv
  face_recognition
  numpy
  os
  glob
  datetime
