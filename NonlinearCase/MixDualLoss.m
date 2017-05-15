function  dobj  =  MixDualLoss(y,Q,alpha,Z,c)

N = length(y);
K = length(c);
if K == 0
		dobj = y'*alpha - norm(alpha)^2/2;
end

DaZ = zeros(N,K);
for k = 1:K
		DaZ(:,k) = alpha .* Z(:,k);
end

dobj = -sum(sum( Q .* (DaZ*diag(c)*DaZ') ) )/2 + y'*alpha  - norm(alpha)^2/2;
