function [best_err, A_out, Z_out, best_m] = Refine_EM(c,Z, y, Q,  K) %round to K

N = length(y);

[c2, ind] = sort(c,'descend');

best_err = 1e300;
T=100;

m_upper = 8;
for m = 1:min(m_upper,length(c))
		%count frequency of distinct patterns of length k
		[Zuni,ia,ic] = unique(Z(:,ind(1:m)),'rows');
		[ic2,ind2] = sort( histc(ic, 1:2^m), 'descend' );
		
		if size(Zuni,1) < K  % #unique patterns less than K
						continue;
		end
		
		%find the K patterns of highest frequencies
		A_init = zeros(N,K);
		for k = 1:K
					S=find(ic==ind2(k));
					alpha_k = ( Q(S,S) + eye(length(S)) ) \ y(S);
					A_init(S,k) = alpha_k;
		end

		%solve with EM initialized by W1
		[alpha2,Z2,A2,RMSE] = EM(y, [], A_init, Q, T);
		
		% test error
		if RMSE < best_err
						best_err = RMSE;
						A_out = A2;
						Z_out = Z2;
						best_m = m;
		end
end
