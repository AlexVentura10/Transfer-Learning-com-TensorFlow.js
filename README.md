# 🧠 Teachable Machine Clone: Transfer Learning com TensorFlow.js

Este projeto demonstra a implementação de um classificador de imagens em tempo real utilizando **Transfer Learning**. O sistema utiliza a inteligência pré-treinada do modelo **MobileNet v3** para extrair características de imagens da webcam e treiná-lo para reconhecer novos objetos personalizados diretamente no navegador.

> **Nota Técnica:** Este projeto foi desenvolvido como parte de um estudo sobre a aplicação de IA em dispositivos de borda (**Edge AI**), conceito fundamental para sistemas de monitoramento como o *Projeto Sentinela*.

---

## 🚀 Funcionalidades

* **Ativação de Webcam:** Interface otimizada para captura de vídeo em tempo real.
* **Coleta de Dados:** Captura e armazenamento de tensores de características para múltiplas classes (ex: Boneco vs. Fusca).
* **Treinamento On-device:** Criação e treinamento de uma rede neural densa (Perceptron) em segundos, sem dependência de servidores externos.
* **Predição em Tempo Real:** Loop de inferência contínuo com exibição de porcentagem de confiança.
* **Gerenciamento de Memória:** Implementação rigorosa de limpeza de tensores (`dispose()`) para evitar vazamentos de memória (Memory Leaks).

---

## 🛠️ Tecnologias e Bibliotecas

* **TensorFlow.js (v3.11.0):** Framework principal para ML em JavaScript.
* **MobileNet v3:** Modelo de rede neural profunda (CNN) truncado para atuar como extrator de características (*feature extractor*).
* **HTML5/CSS3:** Interface responsiva com controle de estado via classes dinâmicas.
* **Vanilla JavaScript:** Lógica de controle assíncrona, manipulação do DOM e gestão de estado do modelo.

---

## 🏗️ Arquitetura do Modelo

O projeto utiliza uma arquitetura híbrida para máxima eficiência:

1.  **Modelo Base (MobileNet v3):** Produz um vetor de **1024 características** (features) para cada frame processado.
2.  **Cabeçalho Personalizado (MLP):**
    * **Camada de Entrada:** Recebe o tensor de shape `[1024]`.
    * **Camada Oculta (Dense):** 128 neurônios com ativação **ReLU**.
    * **Camada de Saída:** Neurônios dinâmicos com ativação **Softmax** para prever probabilidades entre as classes.

---

## 📖 Como Funciona

### 1. Carregamento e "Warm up"
O modelo é carregado do TensorFlow Hub. Para evitar atrasos na primeira inferência, o sistema realiza um "aquecimento" injetando um tensor de zeros assim que o carregamento termina.

### 2. Extração de Features
Ao coletar dados, o frame não é salvo como imagem bruta. Ele é convertido em um Tensor, redimensionado para `224x224`, normalizado e passado pela MobileNet. O que armazenamos são apenas os 1024 números essenciais, o que economiza drasticamente o uso de RAM.

### 3. Treinamento
Utiliza o otimizador **Adam** e a função de perda **Categorical Crossentropy** (ou Binary) para ajustar os pesos da camada personalizada diretamente no hardware do cliente via WebGL.

---

## 💻 Como Executar

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    ```
2.  **Importante:** Utilize um servidor local para evitar erros de política de CORS (ex: Extensão **Live Server** do VS Code).
3.  Abra o `index.html` no navegador.
4.  Clique em **"Ativar Webcam"** e autorize a permissão.
5.  Capture entre 30 e 50 exemplos para cada classe segurando os botões de coleta.
6.  Clique em **"Treinar e Prever"** e observe os resultados no painel de status.

---

## 👨‍💻 Autor

**Alex Mateus da Silva Ventura**
* 🎓 Tecnólogo em Análise e Desenvolvimento de Sistemas (ADS).
* 🤖 Residente em Sistemas Embarcados no **CPQD**.
* 🔭 Interesses: IoT, Inteligência Artificial, Visão Computacional e Edge Computing.

---
Este projeto faz parte do portfólio de estudos em Inteligência Artificial aplicada a sistemas de tempo real.
