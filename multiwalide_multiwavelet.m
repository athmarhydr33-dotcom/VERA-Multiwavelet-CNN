% =========================================================
% multiwalide_multiwavelet.m
% ---------------------------
% 2D Discrete Multi-Wavelet Transform (2D-DMWT)
% Using Critically-Sampled Preprocessing (GHM filter)
%
% INPUT : A  — square image matrix (double, size 2^n x 2^n)
% OUTPUT: B1 — 2D-DMWT coefficient matrix (2N x 2N)
%              Upper-left N x N block = main LL sub-band
%
% Dependencies: generate.m, cspreproc1.m, permutation.m,
%               pppermutesub.m, pppermuterc.m
% =========================================================

function B1 = multiwalide_multiwavelet(A)

[k1, k2] = size(A);
W2 = generate(k1/2, k2/2);

% Critical-sampling preprocessing — rows
AP = cspreproc1(A);
C  = W2 * AP;
CP = permutation(C);

% Critical-sampling preprocessing — columns
D  = CP';
DP = cspreproc1(D);
E  = W2 * DP;
EP = permutation(E);
F  = EP';

% Final sub-band permutation
B1 = pppermutesub(F);
end
