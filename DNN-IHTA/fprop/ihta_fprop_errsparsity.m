function [ ZT, Z, iter, Zerr, avgdiversity_hoyer ] = ihta_fprop_errsparsity(Zorig, X, W, H, t, info, T, maxiter )
%LISTA_FPROP Summary of this function goes here
% [ Z0, Z, C, B ] = lista_fprop( X, We, S, theta, T )
%   Detailed explanation goes here
  B=W*X;
  [Z,~,~]=softthlhalf(B,t*ones(1,size(B,2)));Zk=Z;
  Zerr(1)=norm(Z-Zorig)/norm(Zorig);
  diversity_hoyer = numerosity_hoyer(Z);
  diversity_hoyer(isnan(diversity_hoyer)) = 1;
  avgdiversity_hoyer(1)=mean(diversity_hoyer);
  for iter=1:maxiter
    C=B+H*Zk;
    Z=softthlhalf(C,t*ones(1,size(B,2)));    
    Zerr(1+iter)=norm(Z-Zorig)/norm(Zorig);
    diversity_hoyer = numerosity_hoyer(Z);
    diversity_hoyer(isnan(diversity_hoyer)) = 1;
    avgdiversity_hoyer(1+iter)=mean(diversity_hoyer);
    if norm(Zk-Z)/norm(Zk)<info.thres
        break
    end
    Zk=Z;
    if iter==T
        ZT=Z;
    end
  end
end