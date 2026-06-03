import cv2
import os
import sys

# Definição de caminhos portáveis
cascade_path = os.path.join("cascades", "haarcascade_frontalface_default.xml")
model_path = os.path.join("cascades", "classificadorFisher.yml")

# Verificar se os arquivos necessários existem
if not os.path.exists(cascade_path):
    print(f"Erro: Arquivo Haar Cascade não encontrado em '{cascade_path}'.")
    sys.exit(1)

if not os.path.exists(model_path):
    print(f"Erro: Arquivo do modelo treinado não encontrado em '{model_path}'. Execute 'Algoritmo de treinamento.py' primeiro.")
    sys.exit(1)

detectorFace = cv2.CascadeClassifier(cascade_path)
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read(model_path)

largura, altura = 200, 200
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro: Não foi possível acessar a câmera.")
    sys.exit(1)

# Mapeamento de IDs para classes (rótulos legíveis)
labels = {
    1: "Sem mascara",
    2: "Com mascara"
}

print("Iniciando reconhecimento facial Fisherface. Pressione 'Q' na janela de vídeo para sair...")

try:
    while True:
        conectado, imagem = camera.read()
        if not conectado:
            print("Erro: Falha na leitura do frame da câmera.")
            break

        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))

        for (x, y, l, a) in facesDetectadas:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            
            # Predição do modelo
            id, confianca = reconhecedor.predict(imagemFace)
            
            nome = labels.get(id, f"ID {id}")
            
            # Escrever o rótulo e nível de confiança no frame
            cv2.putText(imagem, nome, (x, y + (a + 30)), font, 1.5, (0, 0, 255), 2)
            cv2.putText(imagem, f"Conf: {round(confianca, 2)}", (x, y + (a + 55)), font, 1.0, (0, 0, 255), 1)

        cv2.imshow("Reconhecimento Facial Fisherface - Pressione Q para sair", imagem)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Liberando recursos do hardware...")
    camera.release()
    cv2.destroyAllWindows()