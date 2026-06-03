import cv2
import numpy as np
import os
import sys

# Garantir que o diretório de saída de fotos existe
os.makedirs('fotos', exist_ok=True)

# Definição de caminhos portáveis para os cascades
cascade_face_path = os.path.join("cascades", "haarcascade_frontalface_default.xml")
cascade_eye_path = os.path.join("cascades", "haarcascade_eye.xml")

classificador = cv2.CascadeClassifier(cascade_face_path)
classificadorOlho = cv2.CascadeClassifier(cascade_eye_path)

if classificador.empty() or classificadorOlho.empty():
    print(f"Erro: Não foi possível carregar os arquivos Haar Cascade. Verifique se estão na pasta '{os.path.abspath('cascades')}'.")
    sys.exit(1)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Erro: Não foi possível acessar a câmera padrão (ID 0). Verifique a conexão.")
    sys.exit(1)

amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador (ex: 1 para sem máscara, 2 para com máscara): ').strip()
largura, altura = 200, 200
print("Iniciando captura das faces. Olhe para a câmera e pressione a tecla 'Q' para registrar cada foto...")

try:
    while True:
        conectado, imagem = camera.read()
        if not conectado:
            print("Erro: Falha na leitura do frame da câmera.")
            break

        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))

        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            regiao = imagem[y:y + a, x:x + l]
            regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)
            for (ox, oy, ol, oa) in olhosDetectados:
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 100:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    caminho_foto = os.path.join("fotos", f"pessoa.{id}.{amostra}.jpg")
                    salvou = cv2.imwrite(caminho_foto, imagemFace)
                    if salvou:
                        print(f"[Foto {amostra} capturada e salva com sucesso em {caminho_foto}]")
                        amostra += 1
                    else:
                        print(f"Erro: Não foi possível salvar a imagem em '{caminho_foto}'. Verifique as permissões de gravação.")

        cv2.imshow("Cadastro de Face - Pressione Q na face detectada", imagem)
        cv2.waitKey(1)
        if amostra >= numeroAmostras + 1:
            break

    print("Todas as faces foram capturadas com sucesso!")

finally:
    print("Liberando recursos do hardware...")
    camera.release()
    cv2.destroyAllWindows()