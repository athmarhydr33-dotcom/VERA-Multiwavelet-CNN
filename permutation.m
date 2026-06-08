% =========================================================
% permutation.m
% --------------
% Row permutation of the wavelet coefficient matrix
%
% INPUT : A — coefficient matrix
% OUTPUT: B — permuted matrix
% =========================================================

function B = permutation(A)

[k1, k2] = size(A);
B = zeros(k1, k2);

k3 = 0;
for i = 1:2:k1/2
    B(i,   :) = A(i   + k3, :);
    B(i+1, :) = A(i+1 + k3, :);
    k3 = k3 + 2;
end

k4 = 0;
for j = k1:-2:k1/2+1
    B(j,   :) = A(j   - k4, :);
    B(j-1, :) = A(j-1 - k4, :);
    k4 = k4 + 2;
end
end
