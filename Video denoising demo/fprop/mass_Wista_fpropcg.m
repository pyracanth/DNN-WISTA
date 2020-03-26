function  [ ZT, Z, iter ] = mass_Wista_fpropcg( X, W, H, info, T, maxiter )
%WISTA_FPROP
  B=W*X;
  t=info.lambda/info.alpha;
  Z=softthl1(B,t);  Zk=0;
  for iter=1:maxiter
    B=B+H*(Z-Zk);
    Z1=Z;Z1(Z1==0)=info.rzero;
     t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
     Zk=Z;
    Z=softthl1(B,t);
     if norm(Z-Zk)/norm(Zk)<1e-4
        break
     end
    if iter==T
        ZT=Z;
    end
  end
end