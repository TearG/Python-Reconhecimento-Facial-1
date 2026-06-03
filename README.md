# 👤 Sistema de Reconhecimento Facial com OpenCV

Este repositório contém uma implementação completa de um pipeline de **Visão Computacional e Reconhecimento Facial** em Python utilizando a biblioteca **OpenCV**. 

O projeto cobre as três etapas fundamentais de um sistema de biometria facial: **Captura de Dados**, **Treinamento de Classificadores** e **Reconhecimento em Tempo Real** utilizando múltiplos algoritmos matemáticos comparativos.

---

## 🎯 Como Funciona o Pipeline

O sistema é dividido em três fases executáveis sequencialmente:

### 1. Captura de Fotos (`Algoritmo de captura.py` ou `Captura.py`)
*   Acessa a câmera (Webcam) do computador em tempo real.
*   Utiliza um classificador **Haar Cascade** (`haarcascade_frontalface_default.xml`) para detectar a presença de faces no vídeo.
*   Recorta a região da face detectada, converte para escala de cinza e redimensiona para um tamanho padrão (ex: 220x220 pixels).
*   Salva até 25 fotos da face na pasta `fotos/` com um ID numérico incremental para treinamento.

### 2. Treinamento (`Algoritmo de treinamento.py`)
*   Lê todas as imagens recortadas salvas na pasta `fotos/`.
*   Mapeia as imagens para os respectivos IDs e treina três modelos preditivos de reconhecimento diferentes.
*   Salva os pesos e os modelos gerados em arquivos YAML dentro da pasta `cascades/`:
    *   `classificadorEigen.yml` (Eigenfaces)
    *   `classificadorFisher.yml` (Fisherfaces)
    *   `classificadorLBPH.yml` (LBPH)

### 3. Reconhecimento Facial em Tempo Real
Existem três scripts correspondentes para testar a predição em tempo real na Webcam usando os classificadores treinados:
*   `Algoritmo de reconhecimento eigenface.py`
*   `Algoritmo de reconhecimento fisherface.py`
*   `Algoritmo de reconhecimento lbph.py`

---

## 💡 Algoritmos Comparados

O projeto compara três abordagens clássicas de reconhecimento facial:

1.  **Eigenfaces (PCA - Principal Component Analysis):**
    *   Focado nas características globais da imagem facial. Ele reduz a dimensionalidade dos dados buscando a direção de maior variância (componentes principais). É sensível a mudanças drásticas de iluminação.
2.  **Fisherfaces (LDA - Linear Discriminant Analysis):**
    *   Busca maximizar a separação entre diferentes classes (pessoas) e minimizar a variação dentro da mesma classe. Apresenta melhor performance do que o Eigenfaces sob variações de iluminação e pose.
3.  **LBPH (Local Binary Patterns Histograms):**
    *   Focado em texturas locais da imagem. Ele compara cada pixel com seus vizinhos em formato binário para criar histogramas locais da pele e características faciais. É o algoritmo mais robusto a variações de luz em ambientes internos.

---

## 🛠️ Tecnologias Utilizadas
*   **Linguagem:** Python
*   **Biblioteca de Visão Computacional:** OpenCV (pacotes `opencv-python` e `opencv-contrib-python`)
*   **Processamento Vetorial:** NumPy

---

## 📁 Estrutura de Arquivos
```text
Reconhecimento Facial/
├── Algoritmo de captura.py                # Captura fotos da webcam e salva no banco de dados
├── Algoritmo de treinamento.py            # Treina os 3 classificadores e exporta os arquivos YAML
├── Algoritmo de reconhecimento eigenface.py  # Teste em tempo real via Eigenface
├── Algoritmo de reconhecimento fisherface.py # Teste em tempo real via Fisherface
├── Algoritmo de reconhecimento lbph.py      # Teste em tempo real via LBPH
├── Captura.py                             # Script alternativo de captura
├── cascades/                              # Armazena os modelos YAML gerados e classificadores Haar
├── fotos/                                 # Banco de imagens das faces capturadas para o treino
└── README.md                              # Documentação do projeto
```

---

## 🚀 Como Executar o Projeto

### Prerrequisitos
*   Possuir o Python 3.x instalado.
*   Ter uma Webcam conectada e configurada no computador.

### 1. Instalar Dependências
No seu terminal, instale os pacotes necessários:
```bash
pip install opencv-python opencv-contrib-python numpy
```
> [!IMPORTANT]
> A biblioteca `opencv-contrib-python` é **obrigatória** porque contém os algoritmos extras de reconhecimento facial (`EigenFaceRecognizer`, `FisherFaceRecognizer` e `LBPHFaceRecognizer`).

### 2. Passo a Passo de Execução

#### Passo 2.1: Capturar Dados
Rode o script de captura. Digite um ID numérico no console quando solicitado (ex: `1` para a primeira pessoa) e olhe para a câmera. O script capturará 25 fotos em escala de cinza e fechará automaticamente.
```bash
python "Algoritmo de captura.py"
```

#### Passo 2.2: Treinar os Modelos
Rode o script de treinamento para ler as fotos da pasta `fotos/` e gerar os classificadores na pasta `cascades/`:
```bash
python "Algoritmo de treinamento.py"
```

#### Passo 2.3: Iniciar o Reconhecimento
Rode qualquer um dos scripts de reconhecimento. O sistema abrirá a câmera, detectará o rosto e exibirá o nome/ID correspondente e a confiança estatística da predição:
```bash
python "Algoritmo de reconhecimento lbph.py"
```

---

*Desenvolvido por Vanessa de Carvalho Faria.*
