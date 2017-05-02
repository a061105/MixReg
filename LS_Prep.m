function [Q, S, QTETy, E] = LS_Prep(y,X,Z)

[N,D] = size(X);
[N,K] = size(Z);
DK = D*K;

E = zeros(N,DK);
for i = 1:N
		rank_one_mat = X(i,:)'*Z(i,:);
		E(i,:) = rank_one_mat(:)';
end

ETE = E'*E;
[Q,S] = eig(ETE);

S = spdiags( diag(S), 0, DK, DK );
QTETy = Q'*E'*y;
