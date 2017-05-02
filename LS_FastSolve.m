function [w,a] = LS_FastSolve(c, Q, S, QTETy, y, E, Tau )

[N,DK] = size(E);
K = length(c);
D = floor(DK/K);

c_expand = reshape( ones(D,1)*c', [DK,1] );
w = Q*inv( S + spdiags(Tau./c_expand,0,DK,DK) )*QTETy;
a = (y-E*w);
