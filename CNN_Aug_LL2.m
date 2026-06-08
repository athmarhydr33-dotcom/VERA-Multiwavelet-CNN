% =========================================================
% CNN_Aug_LL2.m
% --------------
% Step 4 (Baseline) — Lightweight CNN on DWT LL2 Features
%
% INPUT : clahe_LL2_db2/ folder  (from DWT.m)
% OUTPUT: results_DWT_baseline.mat
%
% Identical CNN architecture and training settings
% to CNN_Multiwavelet.m — only features differ.
%
% Random seed: 42
% =========================================================

close all; clear all; clc;
warning off;

rng(42);

datasetPath = 'clahe_LL2_db2';
files       = dir(fullfile(datasetPath,'*.mat'));
numImages   = numel(files);
if isempty(files), error('No .mat files in clahe_LL2_db2/'); end

firstFile       = load(fullfile(datasetPath, files(1).name));
[height, width] = size(firstFile.cA2);
numClasses      = numImages / 20;

imds        = imageDatastore(datasetPath, ...
    'FileExtensions','.mat','ReadFcn',@matRead);
imds.Labels = categorical(repelem(1:numClasses, 20)');

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandXTranslation', [-2 2], ...
    'RandYTranslation', [-2 2], ...
    'RandXScale',       [0.98 1.02], ...
    'RandYScale',       [0.98 1.02]);

augimdsTrain = augmentedImageDatastore([height width 1], imdsTrain, ...
    'DataAugmentation', augmenter);

layers = [
    imageInputLayer([height width 1])
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer; reluLayer
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer; reluLayer
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer; reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer; reluLayer
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer; reluLayer
    globalAveragePooling2dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',    0.001, ...
    'MaxEpochs',           80, ...
    'MiniBatchSize',       16, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      imdsVal, ...
    'ValidationFrequency', 20, ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 70, ...
    'L2Regularization',    0.0001);

net = trainNetwork(augimdsTrain, layers, options);

YPred = classify(net, imdsVal);
YVal  = imdsVal.Labels;
acc   = sum(YPred == YVal) / numel(YVal);
[C,~] = confusionmat(YVal, YPred);
prec  = diag(C) ./ sum(C,1)';
rec   = diag(C) ./ sum(C,2);
f1    = 2*(prec.*rec)./(prec+rec);

fprintf('\n=== DWT Baseline — Validation ===\n');
fprintf('Accuracy  : %.4f (%.2f%%)\n', acc, acc*100);
fprintf('Precision : %.4f\n', mean(prec,'omitnan'));
fprintf('Recall    : %.4f\n', mean(rec, 'omitnan'));
fprintf('F1-score  : %.4f\n', mean(f1,  'omitnan'));

figure; confusionchart(YVal, YPred, ...
    'Title', sprintf('DWT Baseline — Acc: %.2f%%', acc*100));

save('results_DWT_baseline.mat', ...
    'net','acc','C','prec','rec','f1','YPred','YVal');
fprintf('✅ Saved to results_DWT_baseline.mat\n');

function data = matRead(filename)
    in   = load(filename);
    data = reshape(in.cA2, size(in.cA2,1), size(in.cA2,2), 1);
end
