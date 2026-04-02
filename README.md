# 🧠 Teachable Machine Clone: Transfer Learning com TensorFlow.js

Este projeto demonstra a implementação de um classificador de imagens em tempo real utilizando **Transfer Learning**. O sistema utiliza a inteligência pré-treinada do modelo **MobileNet v3** para extrair características de imagens da webcam e treiná-lo para reconhecer novos objetos personalizados diretamente no navegador.

> **Nota Técnica:** Este projeto foi desenvolvido como parte de um estudo sobre a aplicação de IA em dispositivos de borda (**Edge AI**), conceito fundamental para sistemas de monitoramento.

---

## 🚀 Funcionalidades

* **Ativação de Webcam:** Interface otimizada para captura de vídeo em tempo real.
* **Coleta de Dados:** Captura e armazenamento de tensores de características para **3 classes exclusivas**, conforme exigido pelo requisito de negócio.
* **Treinamento On-device:** Criação e treinamento de uma rede neural densa (**Perceptron**) em segundos, sem dependência de servidores externos.
* **Predição em Tempo Real:** Loop de inferência contínuo com exibição de porcentagem de confiança.
* **Gerenciamento de Memória:** Implementação rigorosa de limpeza de tensores (`dispose()`) para evitar vazamentos de memória (*Memory Leaks*).

---

## 🛠️ Tecnologias e Bibliotecas

* **TensorFlow.js:** Framework principal para aprendizado de máquina em JavaScript.
* **MobileNet v3:** Modelo de rede neural profunda (CNN) que atua como extrator de características (*feature extractor*).
* **Ambiente de Desenvolvimento:** Todo o desenvolvimento ocorre no navegador, utilizando **CodePen.io** ou servidor local.
* **Vanilla JavaScript:** Lógica de controle assíncrona, manipulação do DOM e depuração de laços de repetição.

---

## 🏗️ Arquitetura do Modelo

O projeto utiliza uma arquitetura híbrida para máxima eficiência:

1.  **Modelo Base (MobileNet v3):** Atua como os "olhos" do sistema, produzindo um vetor de **1024 características** (*features*) para cada frame processado.
2.  **Cabeçalho Personalizado (MLP):**
    * **Camada de Entrada:** Recebe o tensor de características.
    * **Camada Oculta (Dense):** 128 neurônios com ativação **ReLU**.
    * **Camada de Saída:** Neurônios dinâmicos com ativação **Softmax** para prever probabilidades entre as 3 classes escolhidas.

---

## 📖 Respostas para Reflexão

### Q1. Mecanismo de Reutilização de Conhecimento
O aprendizado por transferência permite reaproveitar um modelo previamente treinado em grandes bases de dados para novas finalidades. Como o modelo já "aprendeu" a extrair características visuais complexas (como linhas, texturas e formas), o processo de reutilizar esse conhecimento torna o novo treinamento muito mais rápido. Além disso, o sistema exige uma quantidade significativamente menor de exemplos do novo objeto para classificá-lo com precisão devido ao treinamento prévio já ocorrido.

### Q2. Eficiência e Execução no Navegador
Esta técnica é ideal para navegadores porque reduz drasticamente o custo computacional e o tempo de treinamento. Em vez de processar milhões de parâmetros do zero, o dispositivo do cliente apenas treina a camada final (cabeçalho). Isso permite que a aplicação seja leve, rápida e execute o treinamento em tempo real com pouquíssimas imagens de amostra, garantindo privacidade ao manter os dados localmente.

### Q3. Arquitetura e o "Cabeçalho de Classificação"
O novo cabeçalho atua como a camada de decisão final, associando os vetores de características da MobileNet às novas classes. O processo é muito mais rápido porque as camadas inferiores estão "congeladas", ou seja, os seus pesos não são alterados. Atualizamos apenas os pesos desta nova camada minimalista, focando o processamento apenas na diferenciação específica entre os objetos escolhidos.

---
Link do roteiro do projeto: https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine?hl=pt-br

## 👨‍💻 Estudante

**Alex Mateus da Silva Ventura**
* 🎓 Tecnólogo em Análise e Desenvolvimento de Sistemas (**ADS**).
* 🤖 Residente em Sistemas Embarcados no **CPQD**.
* 🔭 Interesses: IoT, Inteligência Artificial, Visão Computacional e Edge Computing.

---
*Este projeto faz parte do protocolo de entrega final para a validação do desafio prático de Transfer Learning.*
