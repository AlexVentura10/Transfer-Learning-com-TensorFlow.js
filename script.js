// 1. Constantes de Seleção do DOM e Configurações de Entrada
const STATUS = document.getElementById('status'); // Elemento de texto para dar feedback ao usuário
const VIDEO = document.getElementById('webcam'); // Elemento de vídeo que exibirá a câmera
const ENABLE_CAM_BUTTON = document.getElementById('enableCam'); // Botão para iniciar a webcam
const RESET_BUTTON = document.getElementById('reset'); // Botão para limpar o treinamento
const TRAIN_BUTTON = document.getElementById('train'); // Botão para iniciar o treinamento da rede
const MOBILE_NET_INPUT_WIDTH = 224; // Largura padrão de entrada que a MobileNet espera
const MOBILE_NET_INPUT_HEIGHT = 224; // Altura padrão de entrada que a MobileNet espera
const STOP_DATA_GATHER = -1; // Flag numérica para indicar que não estamos coletando dados
const CLASS_NAMES = []; // Array que guardará os nomes das classes (ex: "Fusca", "Livro")

// 2. Listeners de Eventos: Vincula cliques aos processos do sistema
ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// 3. Gerenciamento da Câmera
function hasGetUserMedia() {
  // Verifica se o navegador possui as APIs necessárias para acessar periféricos de vídeo
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = { video: true, width: 640, height: 480 }; // Define resolução da captura

    // Solicita permissão de acesso à câmera e retorna um fluxo (stream)
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream; // Joga o fluxo da câmera para o elemento <video>
      VIDEO.addEventListener('loadeddata', function() {
        videoPlaying = true; // Flag indicando que o vídeo está pronto para análise
        ENABLE_CAM_BUTTON.classList.add('removed'); // Esconde o botão após ativar
      });
    });
  } else {
    console.warn('Câmera não suportada neste navegador');
  }
}

// 4. Núcleo de Treinamento (Transfer Learning)
async function trainAndPredict() {
  predict = false; // Pausa previsões em tempo real durante o treino para poupar CPU
  
  // Embaralha as entradas e saídas para evitar que o modelo aprenda a ordem dos cliques
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  // Converte labels (0, 1) em tensores para processamento matemático
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  // Transforma labels em One-Hot (ex: Classe 0 vira [1, 0])
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  // Empilha todos os tensores de características coletados em um único bloco de dados
  let inputsAsTensor = tf.stack(trainingDataInputs);
  
  // O Processo de Ajuste (FIT): Treina a rede por 10 rodadas (épocas)
  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true, 
    batchSize: 5, 
    epochs: 10, 
    callbacks: { onEpochEnd: logProgress } // Reporta progresso no console
  });
  
  // Garbage Collection: Libera tensores da memória GPU após o uso
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  
  predict = true; // Reativa a flag de previsão
  predictLoop(); // Inicia o laço de reconhecimento
}

function logProgress(epoch, logs) {
  // Exibe erro (loss) e acurácia no console para debug do desenvolvedor
  console.log('Época: ' + epoch, logs);
}

// 5. Reinicialização do Sistema
function reset() {
  predict = false; // Para o reconhecimento
  examplesCount.length = 0; // Zera contadores visuais
  
  // Varre o array de inputs e deleta cada tensor da memória manualmente
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  
  trainingDataInputs.length = 0; // Esvazia o array de entradas
  trainingDataOutputs.length = 0; // Esvazia o array de labels
  STATUS.innerText = 'Dados resetados';
  
  // Log técnico para verificar se a memória foi realmente limpa
  console.log('Tensores ativos na memória: ' + tf.memory().numTensors);
}

// 6. Configuração dos Botões Dinâmicos de Coleta
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  // Detecta quando o usuário aperta e solta o botão para controlar a coleta
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Povoa o array CLASS_NAMES com os nomes vindos do atributo "data-name" do HTML
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

function gatherDataForClass() {
  // Lê qual classe o botão representa (0 ou 1)
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  // Se o botão for solto, volta para STOP_DATA_GATHER, senão assume o ID da classe
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop(); // Inicia o loop de captura de frames
}

// 7. Variáveis Globais de Estado
let mobilenet = undefined; // Guardará o modelo base carregado
let gatherDataState = STOP_DATA_GATHER; // Estado atual da coleta
let videoPlaying = false; // Estado do hardware de vídeo
let trainingDataInputs = []; // Buffer para as "fotos" (features) processadas
let trainingDataOutputs = []; // Buffer para as labels correspondentes
let examplesCount = []; // Contador de quantas fotos cada classe possui
let predict = false; // Controle do loop de previsão

// 8. Carregamento do Modelo Pré-treinado (Backbone)
async function loadMobileNetFeatureModel() {
  const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  // Faz download e carrega o grafo do modelo MobileNet
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  STATUS.innerText = 'Modelo Base Carregado!';
  
  // Warm up: Faz uma previsão falsa com zeros para inicializar os kernels da GPU
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log("Shape da saída MobileNet:", answer.shape); // Deve ser [1, 1024]
  });
}

loadMobileNetFeatureModel();

// 9. Criação do Modelo de Decisão (Cérebro Customizado)
let model = tf.sequential(); // Modelo em linha (camada após camada)
// Camada Oculta: 128 neurônios que aprendem padrões complexos
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
// Camada de Saída: Neurônios igual ao número de classes (fusca/livro) usando Softmax para probabilidade
model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));

model.summary(); // Imprime a arquitetura no console

// Compilação: Define o "Juiz" (Loss) e o "Professor" (Optimizer)
model.compile({
  optimizer: 'adam', // Ajusta os pesos de forma eficiente
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy', 
  metrics: ['accuracy'] // Queremos monitorar a porcentagem de acertos
});

// 10. Loop de Coleta de Dados
function dataGatherLoop() {
  // Só coleta se o vídeo estiver ativo e o botão estiver pressionado
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let imageFeatures = tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO); // Captura frame da webcam
      // Redimensiona para 224x224
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT, 
          MOBILE_NET_INPUT_WIDTH], true);
      let normalizedTensorFrame = resizedTensorFrame.div(255); // Normaliza pixels de 0-255 para 0-1
      // Passa pela MobileNet para gerar os 1024 números que representam a imagem
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures); // Salva os números da imagem
    trainingDataOutputs.push(gatherDataState); // Salva a qual botão pertencia
    
    // Incrementa contador visual para o usuário
    if (examplesCount[gatherDataState] === undefined) examplesCount[gatherDataState] = 0;
    examplesCount[gatherDataState]++;

    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += CLASS_NAMES[n] + ': ' + (examplesCount[n] || 0) + ' fotos. ';
    }
    // Reexecuta o loop no próximo frame disponível do navegador
    window.requestAnimationFrame(dataGatherLoop);
  }
}

// 11. Loop de Previsão (Reconhecimento)
function predictLoop() {
  if (predict) {
    tf.tidy(function() {
      // Repete o processo de captura e normalização do frame
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [
        MOBILE_NET_INPUT_HEIGHT, 
        MOBILE_NET_INPUT_WIDTH
      ], true);

      // 1. Extrai características com a MobileNet
      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      // 2. Classifica usando o seu modelo treinado
      let prediction = model.predict(imageFeatures).squeeze();
      // 3. Pega o índice do maior valor (quem tem mais probabilidade)
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();

      // Exibe o resultado final na tela com a confiança em %
      STATUS.innerText = 'Previsão: ' + CLASS_NAMES[highestIndex] + 
        ' (' + Math.floor(predictionArray[highestIndex] * 100) + '% de confiança)';
    });

    // Continua prevendo continuamente
    window.requestAnimationFrame(predictLoop);
  }
}