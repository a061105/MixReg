%%Solving the convex MixLasso Objective:
%%
%%   min_{Z\in{0,1}^{N,K}, W\in\R^{K\times D}}  \frac{1}{2N} \sum_{i=1}^N (y_i-z_i^T W x_i)^2 + tau/2||W||_F^2
%%
%%by solving:
%%
%%   min_{M\in\R^{N*N}} max_{a\in\R^N}  
%%      \frac{-1}{2*tau} tr( diag(a)XX'diag(a) M ) + <y,a> - \frac{1}{2}\|a\|^2 + Lambda*\|M\|_S
%%

function [best_round_err] = KernelMixLasso( y, X, Lambda, ker_func, Z0, bash_name )

TOL = 1e-5;
T = 1000;
T2 =1000;
%T_A = 100;
SDP_iter = 200;
SDP_rank = 1;

[N,K0] = size(Z0);
[N,D] = size(X);

'compute kernel matrix'
Q = kernelMat( ker_func, X );
Z = [];
c = [];

%match_tol = 1e-0;
%f = @(W,w1) sqrt( min(sum( (W - ones(size(W,1),1)*w1).^2, 2 ))/D ) <= match_tol;
tol_rate = 0.1;
f = @(Z1,z1,thd) sum( Z1~=(z1*ones(1,size(Z1,2))) ) <= thd;

best_round_err = 1e300;
last_Z = -1;
last_c = -1;
last_obj = 1e300;
eta_rate = 1e-5/N;

for t = 1:T
	
	%find alpha(c)
	K = length(c);
	if K~=0
			M = Z*diag(c)*Z';
	else
			M = zeros(N,N);
	end
	alpha = ( M .* Q + eye(N)) \ y;
	
	%dump info
	dobj = MixDualLoss( y, Q, alpha, Z, c ) + Lambda*sum(c);
	if mod(t,10) == 0
		
		[round_err,Aout,Zout,~] = Refine_EM(c,Z,y,Q,K0);
		if round_err < best_round_err
				best_round_err = round_err;

				%construct predictor
				Xcell = num2cell(X,2);
				pred_funcs = cell(K0,1);
				for k = 1:K0
						pred_funcs{k} = @(x) Aout(:,k)' * cellfun(@(xi) ker_func(xi,x), Xcell);
				end
				%plot
				plot(X,y,'o','color',[0.3,0.3,0.3]);
				hold on;
				x=-1:0.01:1;
				for k = 1:K0
						plot(x, arrayfun(pred_funcs{k},x),'LineWidth',2);
						hold on;
				end
				hold off;
				saveas(gcf,['~/public_html/figures/' bash_name '.pdf'],'pdf');
		end
		['t=' num2str(t) ', d-obj=' num2str(dobj) ', nnz(c)=' num2str(nnz(c)) ', round_loss=' num2str(round_err) ', best_err=' num2str(best_round_err) ', eta=' num2str(eta_rate)]
		%['t=' num2str(t) ', d-obj=' num2str(dobj) ', nnz(c)=' num2str(nnz(c)) ', eta=' num2str(eta_rate)]
	end
	
	if dobj > last_obj + TOL
					'obj increased'
					eta_rate = eta_rate /2;
					c = last_c;
					Z = last_Z;
	else
					last_obj = dobj;
					last_c = c;
					last_Z = Z;
	end
	
	if( t==T )
		break;
	end
	%compute gradient and find greedy direction
	C = diag(alpha)*Q*diag(alpha);
	z = MixMaxCutDense( C, SDP_rank, SDP_iter );
	
	'maxcut done'
	if ~inside( Z, z ) 
		Z = [Z z];
		c = [c;0];
	end
	
	%fully corrective by prox-GD
	k = length(c);
	eta = eta_rate;
	for t2 = 1:T2
			%find alpha(c)
			M = Z*diag(c)*Z';
			alpha = ( M .* Q + eye(N)) \ y;
			%gradient
			grad_c = gradient_c(Z,alpha,Q);
			c = prox( c - eta*grad_c, eta*Lambda );
	end
	
	%shrink c and Z for j:cj=0
	Z = Z(:,c'>TOL);
	c = c(c>TOL);
	
	%'prox-GD done'
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

function grad_c = gradient_c(Z,a,Q)
			
			[N,K] = size(Z);
			Da = diag(a);
			tmp = Da*Q*Da;
			
			grad_c = zeros(K,1);
			for k =1:K
					grad_c(k) = -Z(:,k)'*tmp*Z(:,k)/2;
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
