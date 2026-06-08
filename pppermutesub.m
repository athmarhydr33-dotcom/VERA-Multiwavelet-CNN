% =========================================================
% pppermutesub.m
% ---------------
% Permutation for each of the four sub-bands of matrix A
%
% INPUT : A — 2D-DMWT output matrix
% OUTPUT: B — permuted coefficient matrix
% =========================================================

function B = pppermutesub(A)

[k1, k2] = size(A);
LL = A(1:k1/2,    1:k2/2);
LH = A(1:k1/2,    k2/2+1:k2);
HL = A(k1/2+1:k1, 1:k2/2);
HH = A(k1/2+1:k1, k2/2+1:k2);

B = [pppermuterc(LL), pppermuterc(LH); ...
     pppermuterc(HL), pppermuterc(HH)];
end
