function [alpha, Z, A, RMSE] = EM(y, Z_init, A_init, Q, T)

if isempty(Z_init) && isempty(A_init)
		'Z_init and A_init cannot be both empty'
		return;
end

N = length(y);
if ~isempty(Z_init)
		[~,K] = size(Z_init);
		Z = Z_init;
else
		[~,K] = size(A_init);
		Z = zeros(N,K);
end

alpha = zeros(N,1);

if ~isempty(A_init)
		A = A_init;
else
		A = zeros(N,K);
end

for t = 1:T
		
		if t~=1 || isempty(A_init)
				%M-step
				M = Z*Z';
				alpha = (M.*Q + eye(N)) \ y;
				for k = 1:K
						A(:,k) = alpha.*Z(:,k);
				end
		end
		
		%E-step
		RMSE = 0.0;
		for i=1:N
				pred = A'*Q(:,i);
				[err,k_min] = min(abs( y(i)*ones(K,1)-pred ));
				Z(i,:) = 0;
				Z(i,k_min) = 1;
				
				RMSE = RMSE + err^2;
		end
		RMSE = RMSE/N;
		RMSE = sqrt(RMSE);
		
end

