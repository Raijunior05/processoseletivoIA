import tensorflow as tf
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.h5")
tflite_path = os.path.join(script_dir, "model.tflite")

# 1. Carregar o modelo
model = tf.keras.models.load_model(model_path)

# 2. Configurar conversor para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Aplicar técnica de otimização (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converter o modelo
tflite_model = converter.convert()

# 4. Salvar o resultado
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo otimizado salvo em: {tflite_path}")

# Imprimir os tamanhos para você usar no seu relatório
h5_size = os.path.getsize(model_path) / 1024
tflite_size = os.path.getsize(tflite_path) / 1024
print(f"\nTamanho model.h5: {h5_size:.2f} KB")
print(f"Tamanho model.tflite: {tflite_size:.2f} KB")