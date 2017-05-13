function err = roundErr(y,X,Z,c,K0)

[N,D] = size(X);

if length(c) == 0
				err = norm(y)/sqrt(N);
				return;
end

[c2,ind] = sort(c,'descend');
%Z2 = Z(:,ind(1:min(end,K0)));
Z2 = Z;
[N,K] = size(Z2);
DK = D*K;

E = zeros(N,DK);
for i = 1:N
		rank_one_mat = X(i,:)'*Z2(i,:);
		E(i,:) = rank_one_mat(:)';
end

w = inv(E'*E+1e-5*eye(DK))*E'*y;

err = norm(y-E*w)/sqrt(N);
