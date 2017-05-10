%%Solving the convex MixLasso Objective:
%%
%%   min_{Z\in{0,1}^{N,K}, W\in\R^{K\times D}}  \frac{1}{2N} \sum_{i=1}^N (y_i-z_i^T W x_i)^2 + tau/2||W||_F^2
%%
%%by solving:
%%
%%   min_{M\in\R^{N*N}} max_{a\in\R^N}  
%%      \frac{-1}{2*tau} tr( diag(a)XX'diag(a) M ) + <y,a> - \frac{1}{2}\|a\|^2 + Lambda*\|M\|_S
%%

function [Z,W,c,best_round_err] = MixLasso( y, X, Lambda, Tau, Z0, W0, is_generalized )

TOL = 1e-4;
T = 6;
T2 =1000;
%T_A = 100;
SDP_iter = 1000;
SDP_rank = 10;
Top = 100;

[K0,D] = size(W0);
[N,K0] = size(Z0);
[N,D] = size(X);

Z = [];
c = [];

%match_tol = 1e-0;
%f = @(W,w1) sqrt( min(sum( (W - ones(size(W,1),1)*w1).^2, 2 ))/D ) <= match_tol;
tol_rate = 0.01;
f = @(Z1,z1,thd) sum( Z1~=(z1*ones(1,size(Z1,2))) ) <= thd;

best_round_err = 1e300;
last_Z = -1;
last_c = -1;
last_obj = 1e300;
eta_rate = 1e-2/N;

for t = 1:T
	
	K = length(c);
	if K~=0
		Zc = Z*diag(sqrt(c));
	else
		Zc = zeros(N,1);
		K=1;
	end
	%find a
	[w,a,E] = LS_Solve( y, X, Zc, Tau );
	tmp = reshape(w,[D,K]);
	W = tmp';
	%dump info
	%%
	dobj = MixDualLoss( y, X, Zc, a, Tau ) + Lambda*sum(c);
	pobj = MixPrimalLoss( y, E, w, Tau ) + Lambda*sum(c);
	if mod(t,3) == 0
		if is_generalized
				round_err = Refine_EM_gen(c,Z,y,X,K0);
		else
				round_err = Refine_EM(c,Z,y,X,K0);
		end
		if round_err < best_round_err
				best_round_err = round_err;
				%parameter error (W:K*D, w:[W_{:,1};W_{:,2};...;W_{:,D}])
				%Early Stop coulud be of help
		end
		['t=' num2str(t) ', d-obj=' num2str(dobj) ', p-obj=' num2str(pobj) ', nnz(c)=' num2str(nnz(c)) ', round_loss=' num2str(round_err) ', best_err=' num2str(best_round_err) ', eta=' num2str(eta_rate)]
	end
	if pobj > last_obj + 1e-3
					'obj increased'
					eta_rate = eta_rate /2;
					c = last_c;
					Z = last_Z;
	else
					last_obj = pobj;
	end
	last_c = c;
	last_Z = Z;
	
	if( t==T )
		break;
	end
	%compute gradient and find greedy direction
	A = spdiags(a,0,N,N)*X;
	z = MixMaxCut( A, SDP_rank, SDP_iter);
	'maxcut done'
	
	if ~inside( Z, z ) %&& t<=2
		Z = [Z z];
		c = [c;0];
	end
	
	%fully corrective by prox-GD
	k = length(c);
	
	eta = eta_rate;
	[ETE,ETy,E] = LS_Prep(y,X,Z);
	for t2 = 1:T2
			%Zc = Z*diag(sqrt(c));
			%[w,a,E] = LS_Solve( y, X, Zc, Tau );
			[w,a] = LS_FastSolve(c, ETE, ETy, y,E, Tau);
			grad_c = gradient_c(Z,a,X,Tau);
			c2 = prox( c - eta*grad_c, eta*Lambda );
			delta_c = c2-c;
			%c = c2;
			[tmp,ind] = sort(abs(delta_c),'descend');
			k2 = min(k,Top);
			c(ind(1:k2)) = c2(ind(1:k2));
	end
	%delta_c
	
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	'prox-GD done'
	match = zeros(1,length(c));
	match2 = zeros(1,length(c));
	for k = 1:size(Z0,2)
			match_k = f(Z,Z0(:,k),N*tol_rate);
			match(match_k>0)=k;
			match_k = f(1-Z,Z0(:,k),N*tol_rate);
			match2(match_k>0)=k;
	end
  P	= [match;match2;c'];
	P
end


end

function grad_c = gradient_c(Z,a,X,Tau)
			
			[N,K] = size(Z);
			Da = spdiags(a,0,N,N);
			tmp = Da*X;
			
			grad_c = zeros(K,1);
			for k =1:K
					grad_c(k) = -Z(:,k)'*tmp*tmp'*Z(:,k)/2/Tau;
			end
end
	
function is_inside = inside( Z, z )
	
	is_inside = 0;
	for i = 1:size(Z,2)
		if  all(Z(:,i) == z)
			 is_inside = 1;
		end
	end
end

function x2 = prox( x, lambda )
	
	x2 = x;
	x2(x<=lambda) = 0;
	x2(x>lambda) = x(x>lambda)-lambda;
end
