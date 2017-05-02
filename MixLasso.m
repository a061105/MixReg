%%Solving the convex MixLasso Objective:
%%
%%   min_{Z\in{0,1}^{N,K}, W\in\R^{K\times D}}  \frac{1}{2N} \sum_{i=1}^N (y_i-z_i^T W x_i)^2 + tau/2||W||_F^2
%%
%%by solving:
%%
%%   min_{M\in\R^{N*N}} max_{a\in\R^N}  
%%      \frac{-1}{2*tau} tr( diag(a)XX'diag(a) M ) + <y,a> - \frac{1}{2}\|a\|^2 + Lambda*\|M\|_S
%%

function [Z,W,c] = MixLasso( y, X, Lambda, Tau, Z0 )

TOL = 1e-6;
T = 100;
T2 = 100;
%T_A = 100;
SDP_iter = 100;
SDP_rank = 10;

[N,K0] = size(Z0);
[N,D] = size(X);
Z = [];
c = [];

tol_rate = 0.1;
f = @(Z1,z1,thd) sum( Z1~=(z1*ones(1,size(Z1,2))) ) <= thd;

last_obj = 1e300;
eta = 1e-5;
for t = 1:T
	
	if length(c)~=0
		Zc = Z*diag(sqrt(c));
	else
		Zc = zeros(N,1);
	end
	
	%find a
	[w,a,E] = LS_solve( y, X, Zc, Tau );
	
	%dump info
	%%
	dobj = MixDualLoss( y, X, Zc, a, Tau ) + Lambda*sum(c);
	pobj = MixPrimalLoss( y, E, w, Tau ) + Lambda*sum(c);
	['t=' num2str(t) ', d-obj=' num2str(dobj) ', p-obj=' num2str(pobj) ', nnz(c)=' num2str(nnz(c))]
	
	if( t==T )
		break;
	end
	
	%compute gradient and find greedy direction
	A = spdiags(a,0,N,N)*X;
	z = MixMaxCut(A, SDP_rank, SDP_iter);
	'maxcut done'
	
	if ~inside( Z, z )
		Z = [Z z];
		c = [c;0.0];
	end
	
	%fully corrective by prox-GD
	k = length(c);
	[Q,S,QTETy,E] = LS_prep(y,X,Z);
	for t2 = 1:T2
			[w,a] = LS_FastSolve(c,  Q,S,QTETy,y,E,Tau);
			Da = spdiags(a,0,N,N);
			grad_c = -diag(Z'*Da*X*X'*Da*Z)/2;
			c = prox( c - eta*grad_c, eta*Lambda );
	end
	
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	'prox-GD done'

	match = zeros(1,length(c));
	for k = 1:size(Z0,2)
		match_k = f(Z,Z0(:,k),N*tol_rate);
		match(match_k>0)=k;
	end
	P = [match;c'];
	P
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

function M = compute_M(c,Z)
	
	[n,k] = size(Z);
	M = zeros(n,n);
	for j = 1:k
		M = M + c(j)*Z(:,j)*Z(:,j)';
	end
end

% here A must be a maximizer of dual loss
function grad_M = gradient_M(a, X, Tau)
	
	grad_M = -A*A'/2/Tau;
end

function grad_c = gradient_c(Z,c,A, Tau)
	
	k = length(c);
	grad_c = zeros(k,1);
	%Zc = Z*diag(sqrt(c));
	grad_M = gradient_M(A, Tau);
	for j=1:k
		grad_c(j) = Z(:,j)'*grad_M*Z(:,j);
	end
end

