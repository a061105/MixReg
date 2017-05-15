function Phi = polyExp(X,P)
%Assume X is N by 1

[N,~] = size(X);
Phi = zeros(N,P+1);
Phi(:,1) = ones(N,1);
for p = 1:P
		Phi(:,1+p) = X.^p;
end
