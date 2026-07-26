## 📝 Relatório do Candidato

👤 **Nome Completo:** Raimundo Ferreira do Nascimento Junior

---

### 1️⃣ Resumo da Arquitetura do Modelo

Foi projetada uma CNN (*Convolutional Neural Network*) focada em eficiência paramétrica para cenários de *Edge AI*, composta por 4 estágios convolucionais progressivos (`Conv2D` extraindo 32, 64, 128 e 256 mapas de características, respectivamente). A arquitetura foi desenhada para capturar a hierarquia espacial das imagens RGB (32x32x3) mitigando o *Internal Covariate Shift* através de camadas de `BatchNormalization` acopladas a cada bloco. A redução de dimensionalidade foi tratada via `MaxPooling2D` padronizado.

Na etapa de classificação, os tensores tridimensionais são vetorizados (`Flatten`) e processados por uma densa de 128 neurônios. Para garantir robustez e generalização, adotou-se um `Dropout` agressivo (taxa de 0.5) antes da projeção em *Softmax* (10 classes). O *pipeline* de *Data Augmentation* foi incorporado nativamente ao grafo do Keras via `tf.keras.Sequential` (`RandomFlip`, `RandomRotation` e `RandomZoom`), garantindo que as transformações espaciais ocorressem dinamicamente na CPU durante a alimentação dos *batches*, sem expandir a pegada de memória do *dataset*.

---

### 2️⃣ Bibliotecas Utilizadas

*   **TensorFlow (`v2.15.0`):** Framework principal operando com *backend* Keras para definição declarativa do modelo, compilação de tensores, treinamento otimizado em CPU e exportação unificada.
*   **NumPy (`v1.23+`):** Utilizada para a vetorização avançada e manipulação matricial da saída bruta (vetor de probabilidades) durante o script de inferência.
*   **Módulos Nativos (OS, Sys, IO):** Para roteamento seguro de caminhos de arquivos relativos no ambiente do container e manipulação de *streams* de decodificação UTF-8 via terminal.

---

### 3️⃣ Técnica de Otimização do Modelo

A conversão adotou a técnica de **Post-Training Dynamic Range Quantization** (Quantização de Alcance Dinâmico Pós-Treinamento) via `tf.lite.Optimize.DEFAULT`. Essa técnica atua na camada física do modelo convertendo estaticamente os tensores de pesos (arquitetados originalmente em ponto flutuante de precisão simples `float32`) para inteiros de 8-bits (`int8`). Durante o tempo de inferência, as ativações são quantizadas dinamicamente, permitindo que as operações matemáticas sejam executadas em registradores inteiros das CPUs de dispositivos embarcados. Isso reduz drasticamente a latência e a barreira térmica (*thermal throttling*) sem exigir um conjunto representativo de dados de calibração (*integer-only quantization*).

---

### 4️⃣ Resultados Obtidos

*   🎯 **Acurácia de Validação Final:** `0.7636` (76,36%)
*   📦 **Tamanho `model.h5`:** `6216.09 KB` (~6.2 MB)
*   🚀 **Tamanho `model.tflite`:** `529.25 KB` (~0.5 MB)
*   📉 **Taxa de Compressão:** O modelo foi otimizado atingindo uma redução de tamanho em disco de aproximadamente **11.7x** em relação à arquitetura original.

---

### 5️⃣ Comentários Adicionais (Opcional)

O treinamento de imagens com múltiplos canais (CIFAR-10) estritamente em CPU exige um balanceamento rigoroso entre profundidade arquitetural e *budget* computacional. A limitação a 4 blocos convolucionais — aliada ao monitoramento parametrizado via *Early Stopping* (`patience=5`) — revelou-se uma decisão de engenharia eficaz para maximizar a extração de *features* locais sem gerar sobrecarga excessiva de ciclos de processamento ou estourar o limite de tempo do *runner* no GitHub Actions. Em contextos práticos de Internet das Coisas (IoT), a análise confirmou a tese de que um modelo leve (0.5 MB) entregue via *pipeline CI/CD* com integração contínua gera muito mais valor produtivo do que redes densas e não-otimizadas.

---

### 6️⃣ Exemplo de Inferência

Rodando inferência em 5 amostras usando `model.tflite`:

```text
Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog | real=frog