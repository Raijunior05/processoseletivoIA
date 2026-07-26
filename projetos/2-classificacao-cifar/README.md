# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Raimundo Ferreira do Nascimento Junior

### 1️⃣ Resumo da Arquitetura do Modelo

Foi implementada uma CNN simples, apropriada para Edge AI, composta por 4 blocos convolucionais (Conv2D com 32, 64, 128 e 256 filtros sequencialmente). Cada bloco contém uma camada BatchNormalization (para acelerar e estabilizar o treinamento) e MaxPooling2D (para reduzir a dimensionalidade espacial). O classificador possui uma camada Flatten, seguida de Dropout (0.5) para evitar overfitting, uma camada densa oculta de 128 neurônios e a saída final Softmax de 10 classes. O Data Augmentation foi feito via Sequential Keras integrado diretamente ao input do modelo, aplicando rotação aleatória (RandomRotation), flip horizontal (RandomFlip) e zoom (RandomZoom).

### 2️⃣ Bibliotecas Utilizadas

tensorflow (versão 2.15.0) e módulo interno keras para construção e treinamento do modelo.

numpy (versão 1.23+) para manipulação de arrays e processamento da saída da inferência.

os e io/sys nativos do Python.

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a Dynamic Range Quantization (Quantização de Alcance Dinâmico) por meio da flag tf.lite.Optimize.DEFAULT. Essa técnica reduz de forma estática os pesos do modelo de ponto flutuante de 32-bits (float32) para inteiros de 8-bits (int8), reduzindo o tamanho do modelo em cerca de 4x e agilizando a inferência em CPUs, mantendo a ativação operando em float.

### 4️⃣ Resultados Obtidos

Acurácia de Validação Final: 0.7636

Tamanho model.h5: 6216.09 KB

Tamanho model.tflite: 529.25 KB

### 5️⃣ Comentários Adicionais (Opcional)

Treinar imagens coloridas (CIFAR-10) apenas em CPU mostrou-se computacionalmente denso. A decisão por limitar a 4 blocos com um limitador no Early Stopping (patience=5) foi vital para que o modelo treinasse em tempo aceitável, atingindo o compromisso ideal entre acurácia e o uso de recursos para um ambiente focado em sistemas embarcados.

### 6️⃣ Exemplo de Inferência

6️⃣ Exemplo de Inferência

Rodando inferência em 5 amostras usando model.tflite:
Amostra 1: predito=cat | real=cat
Amostra 2: predito=ship | real=ship
Amostra 3: predito=ship | real=ship
Amostra 4: predito=airplane | real=airplane
Amostra 5: predito=deer | real=frog

Comentário: O modelo teve um desempenho de 80% nessas amostras iniciais. É muito interessante analisar o erro ocorrido na Amostra 5, onde a rede confundiu um sapo (frog) com um cervo (deer). Considerando que as imagens do CIFAR-10 possuem resolução baixíssima (32x32) e animais costumam ser fotografados com fundos verdes/natureza, essa confusão de texturas é bastante compreensível para uma CNN simples otimizada por quantização.