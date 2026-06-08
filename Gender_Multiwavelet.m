% =========================================================
% Gender_Multiwavelet.m
% ----------------------
% Gender Classification using 2D-DMWT Features
% (Proposed Method — Palm / Finger / Fusion)
%
% ── HOW TO USE ───────────────────────────────────────────
%
%   SCENARIO = 'palm'
%   INPUT: VERA_features_DMWT.mat
%          Run Multiwavelet_pre_feature.m with:
%          dataRoot = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw'
%
%   SCENARIO = 'finger'
%   INPUT: VERA_features_DMWT.mat
%          Run Multiwavelet_pre_feature.m with:
%          dataRoot = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw'
%
%   SCENARIO = 'fusion'
%   INPUT: VERA_features_DMWT_palm.mat + VERA_features_DMWT_finger.mat
%          Run Multiwavelet_pre_feature.m on Palm → rename to VERA_features_DMWT_palm.mat
%          Run Multiwavelet_pre_feature.m on Finger → rename to VERA_features_DMWT_finger.mat
%
% OUTPUT: results_gender_DMWT_[scenario].mat
%
% Gender label extracted from VERA folder names:
%   Folders with 'M' → Male (1) | Folders with 'F' → Female (0)
%
% Training Settings (Table 3): Adam, lr=0.001, 80 epochs,
%   batch=16, L2=0.0001, seed=42, 80/20 stratified split
%
% ⚠️  DATASET NOTICE — see README.md
% =========================================================

clear; clc; close all;
warning off;

% ── CHOOSE SCENARIO ──────────────────────────────────────
SCENARIO = 'palm';   % 'palm' | 'finger' | 'fusion'
% ─────────────────────────────────────────────────────────

rng(42);

% ── LOAD DATA ────────────────────────────────────────────
switch SCENARIO
    case 'palm'
        load('VERA_features_DMWT.mat');
        fprintf('Scenario: Palm Vein Gender\n');
    case 'finger'
        load('VERA_features_DMWT.mat');
        fprintf('Scenario: Finger Vein Gender\n');
    case 'fusion'
        palmData   = load('VERA_features_DMWT_palm.mat');
        fingerData = load('VERA_features_DMWT_finger.mat');
        featuresCell  = [palmData.featuresCell,  fingerData.featuresCell];
        labels_person = [palmData.labels_person, fingerData.labels_person];
        fprintf('Scenario: Fusion Gender (%d palm + %d finger = %d total)\n', ...
            length(palmData.featuresCell), length(fingerData.featuresCell), length(featuresCell));
    otherwise
        error('SCENARIO must be: palm | finger | fusion');
end

numImages       = length(featuresCell);
[height, width] = size(featuresCell{1});

% ── EXTRACT GENDER LABELS FROM VERA FOLDER NAMES ─────────
X       = zeros(height, width, 1, numImages, 'single');
genders = zeros(numImages, 1);
for i = 1:numImages
    X(:,:,1,i) = single(featuresCell{i});
    personName = labels_person{i};
    if contains(upper(personName), 'M')
        genders(i) = 1;
    elseif contains(upper(personName), 'F')
        genders(i) = 0;
    else
        error('Cannot determine gender from folder name: %s', personName);
    end
end
Y = categorical(genders);
fprintf('Labels: %d Male | %d Female\n', sum(genders==1), sum(genders==0));

% ── STRATIFIED 80/20 SPLIT ───────────────────────────────
cv      = cvpartition(Y, 'HoldOut', 0.2, 'Stratify', true);
X_train = X(:,:,:, cv.training);  Y_train = Y(cv.training);
X_val   = X(:,:,:, cv.test);      Y_val   = Y(cv.test);
fprintf('Split: %d train | %d val\n', sum(cv.training), sum(cv.test));

% ── DATA AUGMENTATION (training only) ────────────────────
augmenter = imageDataAugmenter( ...
    'RandXReflection',  true, ...
    'RandXTranslation', [-2 2], ...
    'RandYTranslation', [-2 2], ...
    'RandXScale',       [0.98 1.02], ...
    'RandYScale',       [0.98 1.02]);
augimdsTrain = augmentedImageDatastore([height width 1], ...
    X_train, Y_train, 'DataAugmentation', augmenter);

% ── CNN ARCHITECTURE (2-class output) ────────────────────
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
    fullyConnectedLayer(2)
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
YPred    = classify(net, X_val);
accuracy = sum(YPred == Y_val) / numel(Y_val);
[C,~]    = confusionmat(Y_val, YPred);
precision   = diag(C) ./ sum(C,1)';
recall      = diag(C) ./ sum(C,2);
f1          = 2*(precision.*recall)./(precision+recall);
specificity = zeros(2,1);
for c = 1:2
    TP=C(c,c); FP=sum(C(:,c))-TP; FN=sum(C(c,:))-TP; TN=sum(C(:))-TP-FP-FN;
    specificity(c) = TN/(TN+FP);
end

fprintf('\n====== Gender DMWT — %s ======\n', upper(SCENARIO));
fprintf('Overall Accuracy : %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('─────────────────────────────────────\n');
classNames = {'Female','Male'};
for i = 1:2
    fprintf('%s : Prec=%.2f%% | Rec=%.2f%% | F1=%.2f%% | Spec=%.2f%%\n', ...
        classNames{i}, precision(i)*100, recall(i)*100, f1(i)*100, specificity(i)*100);
end
fprintf('─────────────────────────────────────\n');
fprintf('Mean Precision   : %.4f\n', mean(precision,  'omitnan'));
fprintf('Mean Recall      : %.4f\n', mean(recall,     'omitnan'));
fprintf('Mean F1-score    : %.4f\n', mean(f1,         'omitnan'));
fprintf('Mean Specificity : %.4f\n', mean(specificity,'omitnan'));
fprintf('=====================================\n');

figure('Name',['Gender DMWT — ' SCENARIO]);
cm = confusionchart(Y_val, YPred);
cm.Title = sprintf('Gender %s — Acc: %.2f%%', upper(SCENARIO), accuracy*100);
cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

outFile = sprintf('results_gender_DMWT_%s.mat', SCENARIO);
save(outFile,'net','accuracy','C','precision','recall','f1','specificity','YPred','Y_val','SCENARIO');
fprintf('✅ Saved to %s\n', outFile);
