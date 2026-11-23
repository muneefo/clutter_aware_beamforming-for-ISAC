function [V,F] = parte_l2o_proposals(B,M,G,gamma,noiseRatio,weights)
%PARTE_L2O_PROPOSALS Native forward pass for one tone of the frozen Part E model.
% B(:, :, k) contains unit ZF, channel, target and target-null directions.
% M(:, :, k) is the first master's Hermitian pricing matrix for user k.
% G contains normalized channels. gamma and noiseRatio are nominal and unscaled.
[nt,~,K] = size(B);
F = zeros(K,12);
for k = 1:K
    S = B(:,:,k)' * M(:,:,k) * B(:,:,k);
    S = (S+S')/2;
    F(k,1:4) = real(diag(S)).' / max(1e-12,max(abs(S),[],'all'));
    F(k,5:8) = abs(G(:,k)'*B(:,:,k)).^2;
end
F(:,9) = log10(gamma);
F(:,10) = log10(noiseRatio(:))/20;
F(:,11) = mean(F(:,1));
F(:,12) = mean(F(:,5));
coeff = tanh(tanh(F*weights.w0 + weights.w1)*weights.w2 + weights.w3);
coeff(:,3) = coeff(:,3)+1.25;
V = zeros(nt,K,'like',B);
for k = 1:K
    direction = B(:,:,k)*coeff(k,:).';
    length = norm(direction);
    assert(isfinite(length) && length > 1e-12,'Degenerate L2O proposal.');
    V(:,k) = direction/length;
end
end
