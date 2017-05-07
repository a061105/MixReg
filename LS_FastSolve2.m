function [w,a] = LS_FastSolve(c, Q, S_diag, QTETy, y, E, Tau )

[N,DK] = size(E);
K = length(c);
D = floor(DK/K);

c_expand = reshape( ones(D,1)*c', [DK,1] );

Diag = S_diag + Tau./c_expand;
Diag = 1./Diag;

w = Q*( spdiags(Diag,0,DK,DK) )*QTETy

%w = Q*( inv( diag(S_diag)+spdiags(Tau./c_expand,0,DK,DK) ) )*QTETy;
a = (y-E*w);
