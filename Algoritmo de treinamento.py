import cv2
import os
import numpy as np
import sys

# Criar a pasta cascades se não existir
os.makedirs('cascades', exist_ok=True)

# Inicializar os reconhecedores de face do OpenCV
try:
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("Erro: OpenCV face module não está instalado. Instale com: pip install opencv-contrib-python")
    sys.exit(1)

def getImagemComId():
    caminho_fotos = 'fotos'
    if not os.path.exists(caminho_fotos) or not os.path.isdir(caminho_fotos):
        print(f"Erro: O diretório '{caminho_fotos}' não existe. Realize a captura de faces primeiro.")
        return None, None

    arquivos = os.listdir(caminho_fotos)
    # Filtrar apenas por imagens válidas (pessoa.ID.amostra.jpg)
    caminhos = [
        os.path.join(caminho_fotos, f) 
        for f in arquivos 
        if f.startswith('pessoa.') and (f.endswith('.jpg') or f.endswith('.png'))
    ]

    if not caminhos:
        print(f"Erro: Nenhuma imagem de treinamento válida encontrada na pasta '{caminho_fotos}'.")
        return None, None

    faces = []
    ids = []
    
    for caminhoImagem in caminhos:
        imagem = cv2.imread(caminhoImagem)
        if imagem is None:
            print(f"Aviso: Não foi possível ler a imagem {caminhoImagem}. Ignorando.")
            continue
            
        imagemFace = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        try:
            # Pega o ID a partir do nome do arquivo (ex: pessoa.1.15.jpg -> ID = 1)
            id_str = os.path.basename(caminhoImagem).split('.')[1]
            id_val = int(id_str)
        except (IndexError, ValueError) as e:
            print(f"Aviso: Nome de arquivo inválido para o padrão (pessoa.ID.amostra.jpg): {caminhoImagem}. Ignorando.")
            continue
            
        ids.append(id_val)
        faces.append(imagemFace)
        
    return np.array(ids), faces

ids, faces = getImagemComId()

if ids is None or len(ids) == 0:
    print("Erro: Treinamento abortado porque nenhuma face válida foi carregada.")
    sys.exit(1)

print(f"Iniciando treinamento com {len(faces)} faces carregadas...")

try:
    print("Treinando Eigenface...")
    eigenface.train(faces, ids)
    eigenface.write(os.path.join('cascades', 'classificadorEigen.yml'))

    # Fisherface precisa de pelo menos 2 classes diferentes para treinar
    classes_unicas = len(np.unique(ids))
    if classes_unicas >= 2:
        print("Treinando Fisherface...")
        fisherface.train(faces, ids)
        fisherface.write(os.path.join('cascades', 'classificadorFisher.yml'))
    else:
        print("Aviso: Fisherface requer pelo menos 2 classes diferentes (IDs diferentes) para ser treinado. Ignorado.")

    print("Treinando LBPH...")
    lbph.train(faces, ids)
    lbph.write(os.path.join('cascades', 'classificadorLBPH.yml'))

    print("Treinamento concluído com sucesso! Os classificadores YML foram gerados na pasta 'cascades'.")

except Exception as e:
    print(f"Erro durante o treinamento dos classificadores: {str(e)}")
    sys.exit(1)