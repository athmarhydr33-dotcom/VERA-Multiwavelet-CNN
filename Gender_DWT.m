% =========================================================
% Gender_DWT.m
% -------------
% Gender Classification using 2D-DWT LL2 Features
% (Comparative Baseline — Palm / Finger / Fusion)
%
% ── HOW TO USE ───────────────────────────────────────────
%
%   SCENARIO = 'palm'
%   INPUT: raw images from:
%          C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw
%
%   SCENARIO = 'finger'
%   INPUT: raw images from:
%          C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw
%
%   SCENARIO = 'fusion'
%   Set inputDir_palm   → Palm Vein raw folder
%   Set inputDir_finger → Finger Vein raw folder
%   Both will be combined automatically
%
% OUTPUT: results_gender_DWT_[scenario].mat
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

% ── DATASET PATHS ────────────────────────────────────────
inputDir_palm   = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw';
inputDir_finger = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw';
% ─────────────────────────────────────────────────────────

% ── SELECT INPUT DIRECTORY ───────────────────────────────
switch SCENARIO
    case 'palm'
        inputDirs = {inputDir_palm};
        fprintf('Scenario: Palm Vein Gender (DWT baseline)\n');
    case 'finger'
        inputDirs = {inputDir_finger};
        fprintf('Scenario: Finger Vein Gender (DWT baseline)\n');
    case 'fusion'
        inputDirs = {inputDir_palm, inputDir_finger};
        fprintf('Scenario: Fusion Gender (DWT baseline)\n');
    otherwise
        error('SCENARIO must be: palm | finger | fusion');
end

% ── LOAD AND PROCESS IMAGES ──────────────────────────────
numTiles  = [8 8];
clipCLAHE = 0.01;

allFeatures = {};
allGenders  = [];

for d = 1:length(inputDirs)
    inputDir    = inputDirs{d};
    subFolders  = dir(inputDir);
    subFolders  = subFolders([subFolders.isdir] & ...
        ~ismember({subFolders.name},{'.','..'}));

    for s = 1:length(subFolders)
        personFolder = subFolders(s).name;

        % Extract gender from folder name
        if contains(upper(personFolder), 'M')
            gender = 1;
        elseif contains(upper(personFolder), 'F')
            gender = 0;
        else
            continue;
        end

        personPath   = fullfile(inputDir, personFolder);
        sessionDirs  = dir(personPath);
        sessionDirs  = sessionDirs([sessionDirs.isdir] & ...
            ~ismember({sessionDirs.name},{'.','..'}));

        for ses = 1:length(sessionDirs)
            sessionPath = fullfile(personPath, sessionDirs(ses).name);
            imgFiles    = dir(fullfile(sessionPath,'*.png'));

            for f = 1:length(imgFiles)
                img = imread(fullfile(sessionPath, imgFiles(f).name));
                if size(img,3)==3, img = rgb2gray(img); end
                img = imresize(img,[256 256]);
                img = im2uint8(img);

                % CLAHE
                imgCLAHE = adapthisteq(img,'ClipLimit',clipCLAHE,'NumTiles',numTiles);

                % DWT LL2
                [cA,~,~,~]  = dwt2(imgCLAHE,'db2','mode','per');
                [cA2,~,~,~] = dwt2(cA,'db2','mode','per');

                allFeatures{end+1} = cA2;
                allGenders(end+1)  = gender;
            end
        end
    end
end

fprintf('Total: %d images | %d Male | %d Female\n', ...
    length(allFeatures), sum(allGenders==1), sum(allGenders==0));

% ── BUILD 4D ARRAY ───────────────────────────────────────
N               = length(allFeatures);
[height, width] = size(allFeatures{1});
X = zeros(height, width, 1, N, 'single');
for i = 1:N
    img = single(allFeatures{i});
    img = (img - min(img(:))) / (max(img(:)) - min(img(:)) + eps);
    X(:,:,1,i) = img;
end
Y = categorical(allGenders');

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

fprintf('\n====== Gender DWT — %s ======\n', upper(SCENARIO));
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

figure('Name',['Gender DWT — ' SCENARIO]);
cm = confusionchart(Y_val, YPred);
cm.Title = sprintf('Gender DWT %s — Acc: %.2f%%', upper(SCENARIO), accuracy*100);
cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';

outFile = sprintf('results_gender_DWT_%s.mat', SCENARIO);
save(outFile,'net','accuracy','C','precision','recall','f1','specificity','YPred','Y_val','SCENARIO');
fprintf('✅ Saved to %s\n', outFile);
