% =========================================================
% generate.m
% -----------
% Generate the Multi-Wavelet Transformation Matrix
% using GHM filter coefficients
%
% INPUT : k1, k2 — half-dimensions of the input image
% OUTPUT: B      — (2k1) x (2k2) wavelet transformation matrix
% =========================================================

function B = generate(k1, k2)

H0 = [3/(5*sqrt(2)),  4/5;           -1/20,          -3/(10*sqrt(2))];
H1 = [3/(5*sqrt(2)),  0;              9/20,            1/sqrt(2)];
H2 = [0,              0;              9/20,           -3/(10*sqrt(2))];
H3 = [0,              0;             -1/20,            0];
G0 = [-1/20,         -3/(10*sqrt(2)); 1/(10*sqrt(2)),  3/10];
G1 = [ 9/20,         -1/sqrt(2);     -9/(10*sqrt(2)),  0];
G2 = [ 9/20,         -3/(10*sqrt(2)); 9/(10*sqrt(2)), -3/10];
G3 = [-1/20,          0;             -1/(10*sqrt(2)),  0];

H = [H0, H1, H2, H3];
G = [G0, G1, G2, G3];
B = zeros(2*k1, 2*k2);

for i = 1:4:2*k1
    if (i + 7) > (2*k2)
        B(i:i+1,   i:i+3) = H(1:2, 1:4);
        B(i+2:i+3, i:i+3) = G(1:2, 1:4);
        B(i:i+1,   1:4)   = H(1:2, 5:8);
        B(i+2:i+3, 1:4)   = G(1:2, 5:8);
    else
        B(i:i+1,   i:i+7) = H;
        B(i+2:i+3, i:i+7) = G;
    end
end
end
