# Note: This project was originally developed and tested in a Jupyter Notebook (.ipynb). 
# It was later converted into a standalone Python script (.py), with the code split into separate, sequential cells/sections, due to an environment-related error encountered when running the notebook directly.
# The overall logic, architecture, and training pipeline remain unchanged from the original notebook implementation.

# 1. Imports e Configuração de Hardware (AMD RX 6650 XT)
import os
import tensorflow as tf
import keras
import shutil
import kagglehub
import splitfolders
import matplotlib.pyplot as plt
import numpy as np

# Ativação da GPU AMD (Importante para evitar gargalo na CPU)
devices = tf.config.list_physical_devices('GPU')
if devices:
    try:
        for gpu in devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Sucesso! GPU AMD (RX 6650 XT) configurada para treino total.")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print("GPU não detectada. Verifique seu ambiente Conda/DirectML.")

from tensorflow.keras.layers import (Layer, GlobalAveragePooling2D, Dense, Reshape, 
                                     Multiply, Conv2D, BatchNormalization, Activation, 
                                     Add, AveragePooling2D, Concatenate, Input, Dropout)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.regularizers import l2

#=====================================
# 2. Data Generator (Fluxo em Tempo Real - Batch Size 20)
base_path = r"C:\Users\PH\Documents\projetos_Catolica\Relatorio_pibic"
output_folder = os.path.join(base_path, "split_final")

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    os.path.join(output_folder, "train"),
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    os.path.join(output_folder, "val"),
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

test_data = datagen.flow_from_directory(
    os.path.join(output_folder, "test"),
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)
#=====================================
# 3. Arquitetura Inception-MFR Customizada
# Regularização L2 (Weight Decay) aplicada nas camadas Conv2D/Dense.
# OBS: com o otimizador Adam, isso equivale a L2 regularization "acoplada" (não é o
# weight decay desacoplado do AdamW), mas segue a implementação sugerida pelo orientador.
wd = 1e-4

@tf.keras.utils.register_keras_serializable()
class SEBlock(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling2D()
        self.excitation = Dense(self.num_channels // self.ratio, activation='relu',
                                kernel_regularizer=l2(wd))
        self.scale = Dense(self.num_channels, activation='sigmoid',
                           kernel_regularizer=l2(wd))
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = Reshape((1, 1, self.num_channels))(x)
        x = self.excitation(x)
        x = self.scale(x)
        return Multiply()([inputs, x])

def residual_block(x, filters, strides=1):
    shortcut = x
    x = Conv2D(filters, (1, 1), strides=strides, padding='same', kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters * 4, (1, 1), padding='same', kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    if strides != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = Conv2D(filters * 4, (1, 1), strides=strides, padding='same', kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def global_context_block(x):
    gap = GlobalAveragePooling2D()(x)
    gap = Reshape((1, 1, gap.shape[-1]))(gap)
    gap = Conv2D(x.shape[-1], (1, 1), activation='sigmoid', kernel_regularizer=l2(wd))(gap)
    return Multiply()([x, gap])

def inception_module(x, filters):
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu', kernel_regularizer=l2(wd))(x)
    branch3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu', kernel_regularizer=l2(wd))(x)
    branch3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd))(branch3x3)
    branch5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu', kernel_regularizer=l2(wd))(x)
    branch5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu', kernel_regularizer=l2(wd))(branch5x5)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu', kernel_regularizer=l2(wd))(branch_pool)
    return Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])
#=====================================
# 4. Montagem e Configuração do Fine-Tuning Total + Dropout + Weight Decay (L2)
# Duas técnicas de regularização combinadas:
#  - Weight Decay (L2): aplicado via kernel_regularizer nas camadas Conv2D/Dense (ver célula 3, wd=1e-4)
#  - Dropout: aplicado entre o GlobalAveragePooling2D e a camada densa final
DROPOUT_RATE = 0.4

densenet201_base = DenseNet201(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
densenet201_base.trainable = True 

x = densenet201_base.output
x = inception_module(x, [32, 32, 64, 32, 64, 32])
x = residual_block(x, filters=64, strides=2)
x = residual_block(x, filters=64, strides=1)
x = SEBlock()(x)
x = global_context_block(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)
output = Dense(8, activation='softmax', kernel_regularizer=l2(wd))(x)

model = Model(inputs=densenet201_base.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
#=====================================
# 5. Treinamento Inteligente (Scheduler & Checkpoint)
checkpoint_path = os.path.join(base_path, "melhor_modelo_dropout_weight_decay.keras")

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=2, 
    min_lr=1e-6, 
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    checkpoint_path, 
    save_best_only=True, 
    monitor='val_loss', 
    verbose=1
)

print("Iniciando Treino Total (End-to-End) com Dropout + Weight Decay...")
history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data,
    callbacks=[reduce_lr, model_checkpoint],
    verbose=1 
)
#=====================================
# 6. Plot de Resultados (Acurácia e Perda)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino (Acc)')
plt.plot(history.history['val_accuracy'], label='Validação (Acc)')
plt.title('Evolução da Acurácia (Dropout + Weight Decay)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino (Loss)')
plt.plot(history.history['val_loss'], label='Validação (Loss)')
plt.title('Evolução da Perda (Loss) (Dropout + Weight Decay)')
plt.grid(True)
plt.legend()

plt.savefig(os.path.join(base_path, "metricas_finais_dropout_weight_decay.png"))
plt.show()
#=====================================
# 7. Avaliação Detalhada - Teste e Validação juntos (Precision, Recall, F1-Score)
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- Conjunto de TESTE ---
test_data.reset()
y_test_pred_probs = model.predict(test_data)
y_test_pred_classes = np.argmax(y_test_pred_probs, axis=1)
y_test_true = test_data.classes
class_labels = list(test_data.class_indices.keys())
test_report = classification_report(y_test_true, y_test_pred_classes, target_names=class_labels, digits=4)

# --- Conjunto de VALIDAÇÃO ---
validation_data.reset()
y_val_pred_probs = model.predict(validation_data)
y_val_pred_classes = np.argmax(y_val_pred_probs, axis=1)
y_val_true = validation_data.classes
val_report = classification_report(y_val_true, y_val_pred_classes, target_names=class_labels, digits=4)

# --- Impressão conjunta dos dois relatórios ---
relatorio_completo = (
    "Relatório Final (Com Dropout + Weight Decay)\n\n"
    "===================== CONJUNTO DE TESTE =====================\n"
    f"{test_report}\n"
    "=================== CONJUNTO DE VALIDAÇÃO ===================\n"
    f"{val_report}"
)
print(relatorio_completo)

# Salva o relatório conjunto em um único TXT
with open(os.path.join(base_path, "relatorio_dropout_weight_decay.txt"), "w") as f:
    f.write(relatorio_completo)

# --- Matriz de Confusão (apenas do conjunto de TESTE) ---
cm = confusion_matrix(y_test_true, y_test_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão - Teste (Com Dropout + Weight Decay)')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.savefig(os.path.join(base_path, "matriz_confusao_dropout_weight_decay.png"))
plt.show()
