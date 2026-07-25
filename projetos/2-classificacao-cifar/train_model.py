import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

# 1. Carregar dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 2. Normalizar imagens para [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Separar conjunto de validação (10% do treino)
val_split = 0.1
split_idx = int(len(x_train_full) * (1 - val_split))
x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

# 4. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# 5. Construir CNN com 4 blocos + Dropout
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation,
    
    # Bloco 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Bloco 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Bloco 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Bloco 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. EarlyStopping monitorando a perda de validação
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Iniciando o treinamento...")
history = model.fit(
    x_train, y_train,
    epochs=25, # Limitado a 25 para rodar rápido na CPU
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    batch_size=64
)

# 7. Exibir acurácia de validação final
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
print(f"\n--- Acurácia de Validação Final: {val_acc:.4f} ---")

# 8. Salvar o modelo treinado
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.h5")
model.save(model_path)
print(f"Modelo salvo em: {model_path}")