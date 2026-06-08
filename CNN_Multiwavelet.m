% =========================================================
% CNN_Multiwavelet.m
% -------------------
% Step 4 (Proposed) — Lightweight CNN Training
%                     on 2D-DMWT Features
%
% Supports three scenarios:
%   'palm'   → VERA Palm Vein Identification
%   'finger' → VERA Finger Vein Identification
%   'fusion' → VERA Palm-Finger Fusion Identification
%
% Set SCENARIO below before running.
%
% INPUT : VERA_features_DMWT.mat       (palm or finger)
%         VERA_features_DMWT_palm.mat  (fusion only)
%         VERA_features_DMWT_finger.mat(fusion only)
%
% OUTPUT: results_DMWT_[scenario].mat
%
% Training Settings (Table 3):
%   Optimizer        : Adam
%   Learning rate    : 0.001 (×0.1 at epoch 70)
%   Max epochs       : 80
%   Mini-batch size  : 16
%   L2 regularization: 0.0001
%   Augmentation     : RandXReflection, Translation ±2px,
%                      Scale [0.98–1.02] (training only)
%   Split            : 80/20 stratified per identity
%   Random seed      : 42
%
% CNN Architecture (~0.2M parameters):
%   Input → [Conv(3,32)→BN→ReLU]×2 → MaxPool(2)
%         → [Conv(3,64)→BN→ReLU]×2 → MaxPool(2)
%         → [Conv(3,128)→BN→ReLU]×2 → GAP
%         → FC(numClasses) → Softmax
%
% ⚠️  DATASET NOTICE — see README.md
% =========================================================

close all; clear all; clc;
warning off;

% ── CHOOSE SCENARIO ──────────────────────────────────────
SCENARIO = 'palm';   % options: 'palm' | 'finger' | 'fusion'
% ─────────────────────────────────────────────────────────

% ── REPRODUCIBILITY SEED ─────────────────────────────────
rng(42);
% ─────────────────────────────────────────────────────────

% ── LOAD DATA ────────────────────────────────────────────
if strcmp(SCENARIO, 'fusion')
    palmData   = load('VERA_features_DMWT_palm.mat');
    fingerData = load('VERA_features_DMWT_finger.mat');
    featuresCell  = [palmData.featuresCell,  fingerData.featuresCell];
    labels_person = [palmData.labels_person, fingerData.labels_person];
    fprintf('Fusion: %d palm + %d finger = %d total\n', ...
        length(palmData.featuresCell), ...
        length(fingerData.featuresCell), ...
        length(featuresCell));
else
    load('VERA_features_DMWT.mat');
end

numImages       = length(featuresCell);
[height, width] = size(featuresCell{1});
uniquePersons   = unique(labels_person);
numClasses      = length(uniquePersons);

% Build 4D array [H x W x 1 x N]
X = zeros(height, width, 1, numImages, 'single');
Y = zeros(numImages, 1);
idx = 0;
for p = 1:numClasses
    personID  = uniquePersons{p};
    personIdx = find(strcmp(labels_person, personID));
    for i = 1:length(personIdx)
        idx = idx + 1;
        X(:,:,1,idx) = single(featuresCell{personIdx(i)});
        Y(idx) = p;
    end
end
Y = categorical(Y);

% ── STRATIFIED 80/20 SPLIT ───────────────────────────────
% Each identity split independently to preserve class balance.
% Validation images never used for weight optimization.
trainIdx = [];  valIdx = [];
for c = 1:numClasses
    classIdx = find(Y == categorical(c));
    classIdx = classIdx(randperm(length(classIdx)));
    n        = length(classIdx);
    nTrain   = round(0.80 * n);
    trainIdx = [trainIdx; classIdx(1:nTrain)];
    valIdx   = [valIdx;   classIdx(nTrain+1:end)];
end

X_train = X(:,:,:,trainIdx);  Y_train = Y(trainIdx);
X_val   = X(:,:,:,valIdx);    Y_val   = Y(valIdx);

fprintf('Split: %d train / %d val | Classes: %d\n', ...
    length(trainIdx), length(valIdx), numClasses);

% ── DATA AUGMENTATION (training subset only) ─────────────
augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandXTranslation', [-2 2], ...
    'RandYTranslation', [-2 2], ...
    'RandXScale',       [0.98 1.02], ...
    'RandYScale',       [0.98 1.02]);

augimdsTrain = augmentedImageDatastore([height width 1], ...
    X_train, Y_train, 'DataAugmentation', augmenter);

% ── CNN ARCHITECTURE ─────────────────────────────────────
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

% ── TRAINING OPTIONS ─────────────────────────────────────
options = trainingOptions('adam', ...
    'InitialLearnRate',    0.001, ...
    'MaxEpochs',           80, ...
    'MiniBatchSize',       16, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {X_val, Y_val}, ...
    'ValidationFrequency', 20, ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 70, ...
    'L2Regularization',    0.0001);

% ── TRAINING ─────────────────────────────────────────────
net = trainNetwork(augimdsTrain, layers, options);

% ── EVALUATION ───────────────────────────────────────────
YPredTrain = classify(net, X_train);
trainAcc   = sum(YPredTrain == Y_train) / numel(Y_train);
[C_train,~] = confusionmat(Y_train, YPredTrain);
prec_train  = diag(C_train) ./ sum(C_train,1)';
rec_train   = diag(C_train) ./ sum(C_train,2);
f1_train    = 2*(prec_train.*rec_train)./(prec_train+rec_train);

YPred    = classify(net, X_val);
accuracy = sum(YPred == Y_val) / numel(Y_val);
[C,~]    = confusionmat(Y_val, YPred);
precision = diag(C) ./ sum(C,1)';
recall    = diag(C) ./ sum(C,2);
f1        = 2*(precision.*recall)./(precision+recall);

% Specificity per class
specificity = zeros(numClasses,1);
for c = 1:numClasses
    TP = C(c,c);
    FP = sum(C(:,c)) - TP;
    FN = sum(C(c,:)) - TP;
    TN = sum(C(:)) - TP - FP - FN;
    specificity(c) = TN / (TN + FP);
end

fprintf('\n====== DMWT — %s — Validation Results ======\n', upper(SCENARIO));
fprintf('Accuracy    : %.4f (%.2f%%)\n', accuracy,  accuracy*100);
fprintf('Precision   : %.4f\n', mean(precision,   'omitnan'));
fprintf('Recall      : %.4f\n', mean(recall,      'omitnan'));
fprintf('F1-score    : %.4f\n', mean(f1,          'omitnan'));
fprintf('Specificity : %.4f\n', mean(specificity, 'omitnan'));
fprintf('=============================================\n');

% ── CONFUSION MATRICES ───────────────────────────────────
figure('Name',['Training — ' SCENARIO]);
cm1 = confusionchart(Y_train, YPredTrain);
cm1.Title = sprintf('Training %s — Acc: %.2f%%', SCENARIO, trainAcc*100);
cm1.RowSummary = 'row-normalized'; cm1.ColumnSummary = 'column-normalized';

figure('Name',['Validation — ' SCENARIO]);
cm2 = confusionchart(Y_val, YPred);
cm2.Title = sprintf('Validation %s — Acc: %.2f%%', SCENARIO, accuracy*100);
cm2.RowSummary = 'row-normalized'; cm2.ColumnSummary = 'column-normalized';

% ── SAVE ─────────────────────────────────────────────────
outFile = sprintf('results_DMWT_%s.mat', SCENARIO);
save(outFile, 'net', ...
    'trainAcc','C_train','prec_train','rec_train','f1_train', ...
    'accuracy','C','precision','recall','f1','specificity', ...
    'YPred','Y_val','YPredTrain','Y_train','SCENARIO');

fprintf('✅ Saved to %s\n', outFile);
