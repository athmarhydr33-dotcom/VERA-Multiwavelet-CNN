% =========================================================
% CLAHE.m
% --------
% Step 2 — Apply CLAHE contrast enhancement
%
% INPUT : preprocessed_images.mat  (from pre_1.m)
% OUTPUT: clahe_images.mat
%
% CLAHE settings (as reported in the paper):
%   NumTiles  = [8 8]
%   ClipLimit = 0.01
% =========================================================

clear; clc; close all;

load('preprocessed_images.mat', 'allImages', 'allNames');

numTiles  = [8 8];
clipCLAHE = 0.01;

N = numel(allImages);
claheImages = cell(N, 1);

for i = 1:N
    I = allImages{i};
    if ~isa(I, 'uint8'), I = im2uint8(I); end
    claheImages{i} = adapthisteq(I, 'ClipLimit', clipCLAHE, ...
        'NumTiles', numTiles);
end

save('clahe_images.mat', 'claheImages', 'allNames', ...
    'numTiles', 'clipCLAHE', '-v7.3');
fprintf('✅ Done: CLAHE applied to %d images\n', N);
