function Z = randMul(N,K)

Z = zeros(N,K);
for i = 1:N
		k = ceil(rand*K);
		Z(i,k) = 1;
end
