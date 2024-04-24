import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create() # traz a função de reconhecimento Eigenface
fisherface = cv2.face.FisherFaceRecognizer_create() # traz a função de reconhecimento Fisherface
lbph = cv2.face.LBPHFaceRecognizer_create() # traz a função de reconhecimento LBPH

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')] # irá percorrer todas as imagens da pasta fotos criada na captura
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) # transforma as imagens em escala de cinza
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1]) # verifica qual o id do identificador criado na captura
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...") # indicação que está havendo o treinamento, conforme o reconhecedor
eigenface.train(faces, ids)
eigenface.write('cascades\\classificadorEigen.yml') # realiza o treinamento e cria o classificador Eingeface

fisherface.train(faces, ids)
fisherface.write('cascades\\classificadorFisher.yml') # realiza o treinamento e cria o classificador Fisherface

lbph.train(faces, ids)
lbph.write('cascades\\classificadorLBPH.yml') # realiza o treinamento e cria o classificador LBPH

print("Treinamento realizado") # indica que o treinamento foi finalizado
#depois de treinado ele vai gerar os arquivos que vão aparecer no menu ao lado e serão utilizados para o algoritmo de reconhecimento