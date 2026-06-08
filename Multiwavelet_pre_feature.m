% =========================================================
% Multiwavelet_pre_feature.m
% ---------------------------
% Step 3 (Proposed) — 2D-DMWT Feature Extraction Pipeline
%
% Pipeline:
%   1. Image acquisition
%   2. RGB-to-grayscale conversion
%   3. Resize to 256x256
%   4. CLAHE enhancement
%   5. 2D-DMWT decomposition (GHM filter)
%   6. Extract main LL sub-band (128x128)
%   7. Partition LL into four sub-regions
%   8. Average to produce 64x64 feature map (LL2)
%   9. Flatten to 4096-dimensional feature vector
%  10. Save for CNN classification
%
% OUTPUT: VERA_features_DMWT.mat
%   featureMatrix  — [N x 4096]
%   featuresCell   — {N x 1} each 64x64
%   labels_person  — {N x 1}
%   labels_session — {N x 1}
%   labels_filename— {N x 1}
%
% ⚠️  DATASET NOTICE — see README.md
% Dependencies: multiwalide_multiwavelet.m, generate.m,
%               cspreproc1.m, permutation.m,
%               pppermutesub.m, pppermuterc.m
% =========================================================

clear; clc; close all;

% ── SET DATASET PATH ─────────────────────────────────────
% Palm Vein:
dataRoot = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw';

% Finger Vein (uncomment to switch):
% dataRoot = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw';
% ─────────────────────────────────────────────────────────

personFolders = dir(dataRoot);
personFolders = personFolders([personFolders.isdir] & ...
    ~ismember({personFolders.name}, {'.','..'}));

fprintf('Found %d person folders\n', length(personFolders));

featuresCell    = {};
labels_person   = {};
labels_session  = {};
labels_filename = {};
totalFeatures   = 0;
tic;

for p = 1:length(personFolders)
    personName = personFolders(p).name;
    personPath = fullfile(dataRoot, personName);
    sessionFolders = dir(personPath);
    sessionFolders = sessionFolders([sessionFolders.isdir] & ...
        ~ismember({sessionFolders.name}, {'.','..'}));

    for s = 1:length(sessionFolders)
        sessionName = sessionFolders(s).name;
        sessionPath = fullfile(personPath, sessionName);
        imageFiles  = dir(fullfile(sessionPath, '*.png'));

        for f = 1:length(imageFiles)
            img = imread(fullfile(sessionPath, imageFiles(f).name));

            % Step 2: grayscale
            if size(img,3) == 3, img = rgb2gray(img); end

            % Step 3: resize
            img = imresize(img, [256 256]);

            % Step 4: CLAHE
            img_double = double(adapthisteq(img));

            % Step 5: 2D-DMWT
            DMWT_result = multiwalide_multiwavelet(img_double);

            % Step 6: main LL (128x128)
            LL_main = DMWT_result(1:128, 1:128);

            % Steps 7-8: average 4 sub-regions → 64x64
            LL2 = (LL_main(1:64,1:64) + LL_main(1:64,65:128) + ...
                   LL_main(65:128,1:64) + LL_main(65:128,65:128)) / 4;

            totalFeatures = totalFeatures + 1;
            featuresCell{totalFeatures}    = LL2;
            labels_person{totalFeatures}   = personName;
            labels_session{totalFeatures}  = sessionName;
            labels_filename{totalFeatures} = imageFiles(f).name;

            if mod(totalFeatures, 100) == 0
                fprintf('Processed %d images (%.1f s)\n', totalFeatures, toc);
            end
        end
    end
end

fprintf('\n✅ Done: %d features in %.1f s\n', totalFeatures, toc);

% Step 9: flatten to 4096-dim vectors
featureMatrix = zeros(totalFeatures, 4096);
for i = 1:totalFeatures
    featureMatrix(i,:) = featuresCell{i}(:)';
end

% Step 10: save
save('VERA_features_DMWT.mat', ...
    'featureMatrix','featuresCell', ...
    'labels_person','labels_session','labels_filename', '-v7.3');

fprintf('Saved → VERA_features_DMWT.mat\n');
fprintf('Unique persons: %d | Total: %d | Feature: 64x64 = 4096-dim\n', ...
    length(unique(labels_person)), totalFeatures);
