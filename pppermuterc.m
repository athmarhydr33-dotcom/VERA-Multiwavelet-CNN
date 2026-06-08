% =========================================================
% pppermuterc.m
% --------------
% Permutation for columns and rows of a sub-band matrix
%
% INPUT : A — sub-band matrix
% OUTPUT: B — permuted matrix
% =========================================================

function B = pppermuterc(A)

AA  = permutation(A);
APT = AA';
BP  = permutation(APT);
B   = BP';
end
