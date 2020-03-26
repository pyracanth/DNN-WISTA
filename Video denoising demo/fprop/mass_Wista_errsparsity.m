function  [ ZT, Z, iter, Zerr, avgdiversity_hoyer ] = mass_Wista_errsparsity(Zorig, X, W, H, t, info, T, maxiter )
%WISTA_FPROP
  B=W*X;
  Z=softthl1(B,t);  Zk=0;
  Zerr(1)=norm(Z-Zorig)/norm(Zorig);
  diversity_hoyer = numerosity_hoyer(Z);
  diversity_hoyer(isnan(diversity_hoyer)) = 1;
  avgdiversity_hoyer(1)=mean(diversity_hoyer);
  for iter=1:maxiter
    B=B+H*(Z-Zk);
    Z1=Z;Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    Zk=Z;
    Z=softthl1(B,t);
    Zerr(1+iter)=norm(Z-Zorig)/norm(Zorig);
    diversity_hoyer = numerosity_hoyer(Z);
    diversity_hoyer(isnan(diversity_hoyer)) = 1;
    avgdiversity_hoyer(1+iter)=mean(diversity_hoyer);
     if norm(Z-Zk)/norm(Zk)<1e-5
        break
     end
    if iter==T
        ZT=Z;
    end
  end
end