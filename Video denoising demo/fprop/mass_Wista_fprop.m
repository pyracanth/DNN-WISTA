function Z = mass_Wista_fprop( X, W, H, t, T,info)
%LISTA_FPROP
 B=W*X;
 Z=softthl1(B,t);
  for layer=1:T
    Z1=Z;Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    Z=softthl1(B+H*Z,t);
  end
end