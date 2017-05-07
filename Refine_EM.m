function [best_err, W_out, Z_out, best_m] = Refine_EM(c,Z, y,X,  K) %round to K

[N,D] = size(X);

[c2, ind] = sort(c);

best_err = 1e300;
T=100;

m_upper = 9;
for m = 1:min(m_upper,length(c))
		%count frequency of distinct patterns of length k
		[Zuni,ia,ic] = unique(Z(:,ind(1:m)),'rows');
		[ic2,ind2] = sort( histc(ic, 1:2^m), 'descend' );
		
		if size(Zuni,1) < K || ic2(K) < D % #unique patterns less than K
						continue;
		end
		
		%find the K patterns of highest frequencies
		W1 = zeros(K,D);
		for k = 1:K
					S=find(ic==ind2(k));
					wk = inv(X(S,:)'*X(S,:))*X(S,:)'*y(S);
					W1(k,:) = wk';
		end

		%solve with EM initialized by W1
		[W2,Z2] = EM(y,X,W1,T);
		
		% test error
		y2 = sum(Z2 .* (X*W2'),2);
		err = norm(y-y2)/sqrt(N);
		if err < best_err
						best_err = err;
						W_out = W2;
						Z_out = Z2;
						best_m = m;
		end
end
