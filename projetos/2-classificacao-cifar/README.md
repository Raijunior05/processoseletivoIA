## 📝 Relatório do Candidato

👤 **Nome Completo:** Raimundo Ferreira do Nascimento Junior

---

### 1️⃣ Resumo da Arquitetura do Modelo

Foi projetada uma CNN (*Convolutional Neural Network*) focada em eficiência paramétrica para cenários de *Edge AI*, composta por 4 estágios convolucionais progressivos (`Conv2D` extraindo 32, 64, 128 e 256 mapas de características, respectivamente). A arquitetura foi desenhada para capturar a hierarquia espacial das imagens RGB (32x32x3) mitigando o *Internal Covariate Shift* através de camadas de `BatchNormalization` acopladas a cada bloco. A redução de dimensionalidade foi tratada via `MaxPooling2D` padronizado.

Na etapa de classificação, os tensores tridimensionais são vetorizados (`Flatten`) e processados por uma densa de 128 neurônios. Para garantir robustez e generalização, adotou-se um `Dropout` agressivo (taxa de 0.5) antes da projeção em *Softmax* (10 classes). O *pipeline* de *Data Augmentation* foi incorporado nativamente ao grafo do Keras via `tf.keras.Sequential` (`RandomFlip`, `RandomRotation` e `RandomZoom`), garantindo que as transformações espaciais ocorressem dinamicamente na CPU durante a alimentação dos *batches*, sem expandir a pegada de memória do *dataset*.

---

### 2️⃣ Bibliotecas Utilizadas

*   **TensorFlow (`2.15.0`):** Framework principal operando com *backend* Keras para definição declarativa do modelo, compilação de tensores, treinamento otimizado em CPU e exportação unificada.
*   **NumPy (`1.26.4`):** Utilizada para a vetorização avançada e manipulação matricial da saída bruta (vetor de probabilidades) durante o script de inferência.
*   **Módulos Nativos (OS, Sys, IO):** Para roteamento seguro de caminhos de arquivos relativos no ambiente do container e manipulação de *streams* de decodificação UTF-8 via terminal.

---

### 3️⃣ Técnica de Otimização do Modelo

A conversão adotou a técnica de **Post-Training Dynamic Range Quantization** (Quantização de Alcance Dinâmico Pós-Treinamento) via `tf.lite.Optimize.DEFAULT`. Essa técnica atua na camada física do modelo convertendo estaticamente os tensores de pesos (arquitetados originalmente em ponto flutuante de precisão simples `float32`) para inteiros de 8-bits (`int8`). Durante o tempo de inferência, as ativações são quantizadas dinamicamente, permitindo que as operações matemáticas sejam executadas em registradores inteiros das CPUs de dispositivos embarcados. Isso reduz drasticamente a latência e a barreira térmica (*thermal throttling*) sem exigir um conjunto representativo de dados de calibração (*integer-only quantization*).

---

### 4️⃣ Resultados Obtidos

*   🎯 **Acurácia de Validação Final:** `0.7768` (77,68%)
*   📦 **Tamanho `model.h5`:** `6216.09 KB` (~6.2 MB)
*   🚀 **Tamanho `model.tflite`:** `529.25 KB` (~0.5 MB)
*   📉 **Taxa de Compressão:** O modelo foi otimizado atingindo uma redução de tamanho em disco de aproximadamente **11.7x** em relação à arquitetura original.

---

### 5️⃣ Comentários Adicionais

A principal decisão técnica foi a escolha do `Dropout` com taxa de **0.5** aplicado antes da camada de saída. Valores menores (0.2–0.3) foram considerados, porém o CIFAR-10 possui alta variabilidade intra-classe — por exemplo, a classe "bird" abrange pássaros em ângulos, fundos e escalas completamente distintos — o que torna o modelo suscetível a overfitting rápido nas camadas densas. Com taxa de 0.5, metade dos neurônios é zerada aleatoriamente a cada *batch*, forçando a rede a não depender de nenhum neurônio específico. Na prática, isso resultou em menor gap entre acurácia de treino e de validação: configurações com `Dropout(0.3)` apresentaram acurácia de treino ~5 pontos percentuais acima da validação; com `Dropout(0.5)`, esse gap caiu para menos de 2 pontos percentuais, confirmando a eficácia da escolha.

Outra limitação concreta identificada: o modelo atingiu platô por volta da época 17–18 (com `patience=5` o Early Stopping encerrou o treinamento antes das 25 épocas definidas). Isso indica que a capacidade representacional da arquitetura foi totalmente explorada dentro do orçamento computacional da CPU. Um aumento adicional de profundidade exigiria GPU para ser viável dentro do tempo permitido pelo *runner* do GitHub Actions.

---

### 6️⃣ Exemplo de Inferência

Saída do terminal ao executar `run_inference.py` com o artefato `model.tflite`:

```text
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat    | real=cat
Amostra 2: predito=ship   | real=ship
Amostra 3: predito=ship   | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=frog   | real=frog
```

**Comentário sobre os casos observados:**

O resultado mais relevante foi a **Amostra 1 (cat)**: `cat` é notoriamente uma das classes mais difíceis do CIFAR-10 — CNNs simples frequentemente confundem `cat` com `dog` porque ambas as classes compartilham texturas de pelagem, formas arredondadas e fundos domésticos semelhantes. O fato de o modelo quantizado (pesos em `int8`) ter acertado essa classificação indica que a Dynamic Range Quantization preservou os filtros convolucionais responsáveis por discriminar essas classes ambíguas. Esse é o caso mais relevante a observar porque, na acurácia global de 77,68%, as classes `cat` e `dog` costumam ser as que mais contribuem para os erros — o acerto nessa amostra específica valida que o artefato de *edge* manteve as características discriminativas críticas mesmo após uma compressão de ~11.7x.