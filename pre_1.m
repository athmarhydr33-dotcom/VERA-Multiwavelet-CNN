% =========================================================
% pre_1.m
% --------
% Step 1 — Load raw vein images and resize to 256x256
%
% INPUT : raw image folder (set inputDir below)
% OUTPUT: preprocessed_images.mat
%
% ⚠️  DATASET NOTICE — see README.md
% =========================================================

clear; clc; close all;

% ── SET PATH ─────────────────────────────────────────────
% Palm Vein:
inputDir  = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\raw';
outputDir = 'C:\Users\hp\Desktop\VERA-Palmvein\VERA-Palmvein\preprocessed';

% Finger Vein (uncomment to switch):
% inputDir  = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\raw';
% outputDir = 'C:\Users\hp\Desktop\VERA-Fingervein\VERA-Fingervein\preprocessed';
% ─────────────────────────────────────────────────────────

if ~exist(outputDir, 'dir'), mkdir(outputDir); end

imds = imageDatastore(inputDir, 'IncludeSubfolders', true, ...
    'FileExtensions', {'.png'});

allImages = cell(numel(imds.Files), 1);
allNames  = cell(numel(imds.Files), 1);

for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    if size(img, 3) == 3, img = rgb2gray(img); end
    img_resized = imresize(img, [256 256]);
    [~, fn, fe] = fileparts(imds.Files{i});
    imwrite(img_resized, fullfile(outputDir, [fn, fe]));
    allImages{i} = img_resized;
    allNames{i}  = fn;
end

save('preprocessed_images.mat', 'allImages', 'allNames', '-v7.3');
fprintf('✅ Done: %d images → preprocessed_images.mat\n', numel(allImages));
