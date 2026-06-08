% =========================================================
% cspreproc1.m
% -------------
% Critically-Sampled Preprocessing — First Order Type
%
% INPUT : A — input matrix
% OUTPUT: B — preprocessed matrix
% =========================================================

function B = cspreproc1(A)

[k1, k2] = size(A);
B = zeros(k1, k2);

for i = 1:k1
    if i == 1
        B(i,:) = 0.373615 * A(i,:) + 0.11086198 * A(i+1,:);
    elseif (i/2) == round(i/2)
        B(i,:) = (sqrt(2) - 1) * A(i,:);
    else
        B(i,:) = 0.373615 * A(i,:) + ...
                 0.11086198 * (A(i+1,:) + A(i-1,:));
    end
end
end
