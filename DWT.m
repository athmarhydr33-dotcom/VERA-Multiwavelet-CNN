% =========================================================
% DWT.m
% ------
% Step 3 (Baseline) — 2D-DWT Feature Extraction
%
% INPUT : clahe_images.mat  (from CLAHE.m)
% OUTPUT: clahe_wavelet_db2/ — LL1 features (.mat)
%         clahe_LL2_db2/     — LL2 features (.mat)
%
% Wavelet: db2, periodic boundary
% =========================================================

close all; clear all; clc;
warning off;

load('clahe_images.mat', 'claheImages');

if ~exist('clahe_wavelet_db2','dir'), mkdir('clahe_wavelet_db2'); end
if ~exist('clahe_LL2_db2',    'dir'), mkdir('clahe_LL2_db2');     end

for i = 1:numel(claheImages)
    a = claheImages{i};

    [cA,  ~,~,~] = dwt2(a,  'db2','mode','per');
    save(sprintf('clahe_wavelet_db2/image_%04d.mat',i), 'cA');

    [cA2, ~,~,~] = dwt2(cA, 'db2','mode','per');
    save(sprintf('clahe_LL2_db2/image_%04d.mat',i), 'cA2');
end

fprintf('✅ DWT baseline done: %d images\n', numel(claheImages));
