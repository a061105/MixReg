function pobj = MixPrimalLoss(y, Q, a, Z, c )

N = length(y);
K = length(c);
if K == 0
		pobj = norm(y)^2/2;
		return;
end

DaZ = zeros(N,K);
for k = 1:K
		DaZ(:,k) = a .* Z(:,k);
end

y2 = sum(Q*DaZ*diag(sqrt(c)),2);

pobj = norm(y-y2)^2/2 + sum(sum(Q.*(DaZ*diag(c)*DaZ')))/2;
