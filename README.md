# 🩺 GI Disease Classification with DenseNet201 + Attention Mechanisms

> Undergraduate research project (PIBIC), exploring deep learning architectures — Inception modules, Residual blocks, Squeeze-and-Excitation attention, and Global Context blocks — for multiclass gastrointestinal (GI) disease classification from endoscopic images, with a comparative study of regularization techniques (Weight Decay and Dropout).

🇧🇷 [Leia em Português](#-classificação-de-doenças-gastrointestinais-com-densenet201--mecanismos-de-atenção)

---

## 🇬🇧 English

### Overview

This repository contains the code developed during a PIBIC (undergraduate scientific research program, Brazil) project focused on classifying **8 gastrointestinal (GI) disease/anatomical categories** from endoscopic images. The work replicates and extends the architecture proposed in:

> Hosny, M., Elgendy, I. A., & Albashrawi, M. A. (2026). *Beyond transfer learning: Attention-enhanced deep learning framework for multiclass gastrointestinal disease classification.* Expert Systems With Applications, 295, 128852. https://doi.org/10.1016/j.eswa.2025.128852

The reference paper introduces **IRA-DenseNet201**, combining a **DenseNet201** backbone (pretrained on ImageNet) with custom modules:

- **Inception module** — multi-scale feature extraction via parallel convolution branches (1x1, 3x3, 5x5, pooling).
- **Residual blocks** — two consecutive bottleneck-style residual connections for deeper, more stable feature refinement.
- **Squeeze-and-Excitation (SE) Block** — channel-wise attention to recalibrate feature importance (the paper also compares CBAM, ECANet, Self-Attention, and Triplet Attention, with SE performing best).
- **Global Context Block** — global feature recalibration before classification.

On top of this shared architecture, this project evaluates the effect of **two regularization techniques**, individually and combined, which are not part of the original paper:

| Variant | Description |
|---|---|
| **Baseline** | Full architecture, replicating the reference paper's design. |
| **Weight Decay (L2)** | `kernel_regularizer=l2(1e-4)` applied to convolutional and dense layers. |
| **Dropout** | `Dropout(0.4)` applied between Global Average Pooling and the final Dense layer. |
| **Dropout + Weight Decay** | Both techniques combined in the same model. |

### Repository Structure

```
├── notebooks/
│   ├── PibicMadeiro3.ipynb              # Baseline (replicates the reference architecture)
│   ├── PibicMadeiroWD.ipynb             # Baseline + Weight Decay (L2)
│   ├── PibicMadeiroDropout.ipynb        # Baseline + Dropout
│   └── PibicMadeiroDropoutWD.ipynb      # Baseline + Dropout + Weight Decay
├── results/                             # Classification reports, confusion matrices, training curves
└── README.md
```

### Dataset

- **8 classes**, spanning anatomical landmarks and pathological findings (as in the reference paper's Kvasir-based setup): dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, and ulcerative-colitis.
- Images resized to `224x224`, RGB, pixel values rescaled to `[0, 1]`.
- Split into `train` / `val` / `test` folders (`ImageDataGenerator.flow_from_directory`): 2800 training images, 600 validation images, 600 test images.

> Note: dataset not included in this repository. Update `base_path` / `output_folder` in each notebook to point to your local data split.

### Model Training

All variants share the same training configuration for a fair comparison:

- Optimizer: `Adam(learning_rate=1e-4)`
- Loss: `categorical_crossentropy`
- Callbacks: `ReduceLROnPlateau` (factor 0.5, patience 2) and `ModelCheckpoint` (best `val_loss`)
- Epochs: 20
- Batch size: 20

### Evaluation

Each notebook reports:

- **Training curves** (accuracy and loss, train vs. validation).
- **Classification report** (precision, recall, F1-score) on the test set — and, in later notebooks, on the validation set as well.
- **Confusion matrix** on the test set.

### Requirements

```
tensorflow
keras
scikit-learn
matplotlib
seaborn
numpy
splitfolders
kagglehub
```

### How to Run

1. Clone the repository and install the dependencies (`pip install -r requirements.txt`).
2. Organize your dataset into `train/`, `val/`, and `test/` subfolders (one folder per class).
3. Update the `base_path` variable at the top of each notebook.
4. Run the notebooks in order, or independently — each one is self-contained.

### Acknowledgements

This project was developed as part of a PIBIC undergraduate research program, replicating and extending the IRA-DenseNet201 architecture proposed by Hosny et al. (2026).

### License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🇧🇷 Classificação de Doenças Gastrointestinais com DenseNet201 + Mecanismos de Atenção

🇬🇧 [Read in English](#-english)

### Visão Geral

Este repositório contém o código desenvolvido durante um projeto de PIBIC (Programa Institucional de Bolsas de Iniciação Científica), focado na classificação de **8 categorias de doenças/estruturas gastrointestinais (GI)** a partir de imagens endoscópicas. O trabalho replica e estende a arquitetura proposta em:

> Hosny, M., Elgendy, I. A., & Albashrawi, M. A. (2026). *Beyond transfer learning: Attention-enhanced deep learning framework for multiclass gastrointestinal disease classification.* Expert Systems With Applications, 295, 128852. https://doi.org/10.1016/j.eswa.2025.128852

O artigo de referência propõe a **IRA-DenseNet201**, que combina um backbone **DenseNet201** (pré-treinado no ImageNet) com módulos customizados:

- **Módulo Inception** — extração de características em múltiplas escalas, via ramos paralelos de convolução (1x1, 3x3, 5x5, pooling).
- **Blocos Residuais** — duas conexões residuais consecutivas no estilo *bottleneck*, para refinamento mais profundo e estável das features.
- **Squeeze-and-Excitation (SE) Block** — atenção por canal, recalibrando a importância de cada característica (o artigo também compara CBAM, ECANet, Self-Attention e Triplet Attention, com o SE obtendo o melhor desempenho).
- **Global Context Block** — recalibração global das características antes da classificação final.

Sobre essa arquitetura compartilhada, este projeto avalia o efeito de **duas técnicas de regularização**, isoladas e combinadas, que não fazem parte do artigo original:

| Variante | Descrição |
|---|---|
| **Baseline** | Arquitetura completa, replicando o design do artigo de referência. |
| **Weight Decay (L2)** | `kernel_regularizer=l2(1e-4)` aplicado nas camadas convolucionais e densas. |
| **Dropout** | `Dropout(0.4)` aplicado entre o Global Average Pooling e a camada Dense final. |
| **Dropout + Weight Decay** | As duas técnicas combinadas no mesmo modelo. |

### Estrutura do Repositório

```
├── notebooks/
│   ├── PibicMadeiro3.ipynb              # Baseline (replica a arquitetura de referência)
│   ├── PibicMadeiroWD.ipynb             # Baseline + Weight Decay (L2)
│   ├── PibicMadeiroDropout.ipynb        # Baseline + Dropout
│   └── PibicMadeiroDropoutWD.ipynb      # Baseline + Dropout + Weight Decay
├── results/                             # Relatórios de classificação, matrizes de confusão, curvas de treino
└── README.md
```

### Dataset

- **8 classes**, abrangendo marcos anatômicos e achados patológicos (conforme o setup baseado no Kvasir do artigo de referência): dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps e ulcerative-colitis.
- Imagens redimensionadas para `224x224`, RGB, valores de pixel normalizados para `[0, 1]`.
- Divididas em pastas `train` / `val` / `test` (`ImageDataGenerator.flow_from_directory`): 2800 imagens de treino, 600 de validação, 600 de teste.

> Observação: o dataset não está incluído neste repositório. Atualize as variáveis `base_path` / `output_folder` em cada notebook para apontar para o seu split local dos dados.

### Treinamento dos Modelos

Todas as variantes compartilham a mesma configuração de treino, para permitir uma comparação justa:

- Otimizador: `Adam(learning_rate=1e-4)`
- Função de perda: `categorical_crossentropy`
- Callbacks: `ReduceLROnPlateau` (fator 0.5, paciência 2) e `ModelCheckpoint` (melhor `val_loss`)
- Épocas: 20
- Tamanho do batch: 20

### Avaliação

Cada notebook reporta:

- **Curvas de treinamento** (acurácia e perda, treino vs. validação).
- **Relatório de classificação** (precision, recall, F1-score) no conjunto de teste — e, nos notebooks mais recentes, também no conjunto de validação.
- **Matriz de confusão** no conjunto de teste.

### Requisitos

```
tensorflow
keras
scikit-learn
matplotlib
seaborn
numpy
splitfolders
kagglehub
```

### Como Executar

1. Clone o repositório e instale as dependências (`pip install -r requirements.txt`).
2. Organize seu dataset em subpastas `train/`, `val/` e `test/` (uma pasta por classe).
3. Atualize a variável `base_path` no início de cada notebook.
4. Execute os notebooks na ordem, ou de forma independente — cada um é autocontido.

### Agradecimentos

Este projeto foi desenvolvido como parte de um programa de Iniciação Científica (PIBIC), sob orientação do **Prof. Madeiro**, replicando e estendendo a arquitetura IRA-DenseNet201 proposta por Hosny et al. (2026).

### Licença

Este projeto está licenciado sob a Licença MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.
