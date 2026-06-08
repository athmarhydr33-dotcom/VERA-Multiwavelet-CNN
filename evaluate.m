% =========================================================
% evaluate.m
% -----------
% Step 5 — Full Evaluation Metrics
%
% Computes: Accuracy, Precision, Recall, F1-score,
%           Specificity (per class and mean)
%
% INPUT : any results_*.mat file
% OUTPUT: printed metrics table
% =========================================================

clear; clc;

% ── CHOOSE RESULTS FILE ──────────────────────────────────
resultsFile = 'results_DMWT_palm.mat';
% Other options:
%   'results_DMWT_finger.mat'
%   'results_DMWT_fusion.mat'
%   'results_DWT_baseline.mat'
% ─────────────────────────────────────────────────────────

S = load(resultsFile);

if isfield(S,'accuracy'), acc = S.accuracy;
elseif isfield(S,'acc'),  acc = S.acc; end

if isfield(S,'C'), C = S.C;
elseif isfield(S,'C_val'), C = S.C_val; end

numClasses  = size(C,1);
precision   = diag(C) ./ sum(C,1)';
recall      = diag(C) ./ sum(C,2);
f1          = 2*(precision.*recall)./(precision+recall);
specificity = zeros(numClasses,1);

for c = 1:numClasses
    TP = C(c,c);
    FP = sum(C(:,c)) - TP;
    FN = sum(C(c,:)) - TP;
    TN = sum(C(:)) - TP - FP - FN;
    specificity(c) = TN / (TN + FP);
end

fprintf('\n════════════════════════════════════════\n');
fprintf('  Results: %s\n', resultsFile);
fprintf('════════════════════════════════════════\n');
fprintf('  Accuracy    : %.4f  (%.2f%%)\n', acc,  acc*100);
fprintf('  Precision   : %.4f\n', mean(precision,   'omitnan'));
fprintf('  Recall      : %.4f\n', mean(recall,      'omitnan'));
fprintf('  F1-score    : %.4f\n', mean(f1,          'omitnan'));
fprintf('  Specificity : %.4f\n', mean(specificity, 'omitnan'));
fprintf('════════════════════════════════════════\n');
