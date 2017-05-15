function Q = kernelMat( ker_func, X)

[N,D] = size(X);

Q = zeros(N,N);

for i = 1:N
		for j = 1:N
				Q(i,j) = ker_func(X(i,:),X(j,:));
		end
end

