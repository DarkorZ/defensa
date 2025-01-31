% Limpiar el entorno de trabajo
clc; clear; close all;

% ==============================
% 1. Preparación de datos
% ==============================
numImages = 200;           % Número total de imágenes (entrenamiento)
imageSize = [90, 60];      % Tamaño estándar de las imágenes (90x60 píxeles)
numFeatures = prod(imageSize); % Número de características por imagen (5400 píxeles)
Xtrain = zeros(numImages, numFeatures);  % Matriz para almacenar características
Ytrain = zeros(numImages, 1);  % Vector para almacenar etiquetas

% ==============================
% 2. Procesar imágenes de "male"
% ==============================
for i = 1:100
    fileName = sprintf("male%d.jpg", i); % Generar nombre del archivo
    RGB = imread(fileName);             % Leer imagen
    RGB = imresize(RGB, imageSize);     % Redimensionar a 90x60 píxeles
    I = rgb2gray(RGB);                  % Convertir a escala de grises
    Xtrain(i, :) = I(:)';               % Aplanar y almacenar en Xtrain
    Ytrain(i) = 1;                      % Etiqueta: hombre (1)
end

% ==============================
% 3. Procesar imágenes de "female"
% ==============================
for i = 1:100
    fileName = sprintf("female%d.jpg", i); % Generar nombre del archivo
    RGB = imread(fileName);               % Leer imagen
    RGB = imresize(RGB, imageSize);       % Redimensionar a 90x60 píxeles
    I = rgb2gray(RGB);                    % Convertir a escala de grises
    Xtrain(i + 100, :) = I(:)';           % Aplanar y almacenar en Xtrain
    Ytrain(i + 100) = -1;                 % Etiqueta: mujer (-1)
end

% ==============================
% 4. Inicialización y entrenamiento del perceptrón
% ==============================
W = zeros(1, numFeatures + 1); % Vector de pesos inicializado en ceros (incluye bias)
w_values = perceptronLearning(Xtrain, Ytrain, W'); % Entrenar el modelo

% ==============================
% 5. Preparación de datos de prueba
% ==============================
numImages_2 = 44;          % Número total de imágenes de prueba
Xtest = zeros(numImages_2, numFeatures); % Matriz de características de prueba
Ytest = zeros(numImages_2, 1);  % Vector de etiquetas de prueba

% ==============================
% 6. Procesar imágenes de prueba de hombres
% ==============================
for i = 1:20
    fileName2 = sprintf("m_test%d.jpg", i); % Generar nombre del archivo
    RGB2 = imread(fileName2);               % Leer imagen
    RGB2 = imresize(RGB2, imageSize);       % Redimensionar
    I2 = rgb2gray(RGB2);                    % Convertir a escala de grises
    Xtest(i, :) = I2(:)';                   % Aplanar y almacenar en Xtest
    Ytest(i) = 1;                           % Etiqueta: hombre (1)
end

% ==============================
% 7. Procesar imágenes de prueba de mujeres
% ==============================
for i = 1:24
    fileName2 = sprintf("f_test%d.jpg", i); % Generar nombre del archivo
    RGB2 = imread(fileName2);               % Leer imagen
    RGB2 = imresize(RGB2, imageSize);       % Redimensionar
    I2 = rgb2gray(RGB2);                    % Convertir a escala de grises
    Xtest(i + 20, :) = I2(:)';              % Aplanar y almacenar en Xtest
    Ytest(i + 20) = -1;                     % Etiqueta: mujer (-1)
end

% ==============================
% 8. Clasificación de imágenes de prueba
% ==============================
w_final=w_values(end,:); % Escoger el último de los pesos
 = perceptronOutput(Xtest, w_final'); % Clasificar imágenes de prueba

% ==============================
% 9. Visualización de resultados
% ==============================
numMuestra = 20; % Número de imágenes a mostrar
randomIndices = randperm(size(Xtest, 1), numMuestra); % Índices aleatorios

figure;
for i = 1:numMuestra
    idx = randomIndices(i);                  % Índice de la imagen a mostrar
    img = reshape(Xtest(idx, :), imageSize); % Reconstruir la imagen desde el vector
    prediccion = perceptronOutput(Xtest(idx, :), w_final'); % Predicción del modelo
    etiqueta = 'hombre';                     % Etiqueta por defecto
    if prediccion < 0
        etiqueta = 'mujer';                  % Cambiar etiqueta si es mujer
    end
    subplot(4, 5, i);                        % Crear subgráfica
    imshow(uint8(img));                      % Mostrar imagen
    title(etiqueta);                         % Título con la predicción
end

% Título general para las subgráficas
sgtitle('Muestra aleatoria de imágenes con etiquetas predichas');




