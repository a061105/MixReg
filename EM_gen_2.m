function [err,W,Z]=EM_gen(k,max_iter,epsilon,tau,verbose,X,Y,W_init)
    % W: output weight guess d*k
    % Z: output hidden class n*k
    % met1:parameter error
    % met2:RMSE
    % max_iter: maximum iteration
    % tau : tikhonov regularization
    % epsilon: for likelihood
    % X: follow N(0,I) n*d
    % Y: sum(Z_t.*(X*W_t),2) t:for true
    % W_init: initial guess of W 
     rng('shuffle','twister');
     [n,d]=size(X);
     Z=zeros(n,k);
     if nargin < 8
        W_init = randn(d,k); 
     end
     W=W_init;
     M = (dec2bin(0:(2^k)-1)=='1'); % M: 2^k*k
     Y_aug=repmat(Y,1,2^k);     
     for i=1:max_iter

				% E-step
        [~,ind]=min(abs(Y_aug-X*W*M'),[],2);
        Z = M(ind,:);
        
				% M-step
        X_aug = repmat(X,1,k).*repelem(Z,1,d);
        W_flat = inv(X_aug'*X_aug + tau*eye(d*k))*X_aug'*Y;
        W=reshape(W_flat,[d,k]);
        
				err= norm(sum(Z.*(X*W),2)-Y)/sqrt(n); 
        if verbose == 1
            fprintf('At iter=%d,RMSE=%f\n',i,err);
        end
        if  err < epsilon
            fprintf('RMSE small, quit\n');
            break;
        end    
     end
end
