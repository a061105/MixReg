function [w,a,E] = LS_solve( y, X, Zc, Tau )

[N,D] = size(X);
[N,K] = size(Zc);
DK = D*K;

E = zeros(N,DK);
for i = 1:N
	rank_one_mat = X(i,:)'*Zc(i,:);
	E(i,:) = rank_one_mat(:)';
end

w = inv(E'*E + Tau*eye(DK) )*E'*y; %DK*1
a = (y-E*w);
