function [W_out, Z] = EM(y,X,W_init,T)

[N,D] = size(X);
[K,D] = size(W_init);

W = W_init;
assgin = zeros(N,1);

for t = 1:T
	
	%E-Step
	for i = 1:N
			pred = W*X(i,:)';
			[tmp,ind] = min(abs(y(i)*ones(K,1)-pred));
			assign(i) = ind;
	end

	%M-Step
	for k = 1:K
			S = find(assign==k);
			wk = inv(X(S,:)'*X(S,:))*X(S,:)'*y(S);
			W(k,:) = wk';
	end
end

W_out = W;
Z = zeros(N,K);
for i =1:N
		Z(i,assign(i)) = 1;
end
