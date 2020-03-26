function [ Z, C, B ] = Wista_fpropLnet( X, W, H, t, T ,info)
%LISTA_FPROP Summary of this function goes here
% [ Z0, Z, C, B ] = lista_fprop( X, We, S, theta, T )
%   Detailed explanation goes here
  B=W*X;
  C=zeros(numel(B),T);
  Z=zeros(numel(B),T+1);
  Z(:,1)=h_theta(B,t);
  for iter=1:T
    C(:,iter)=B+H*Z(:,iter);
    Z1=Z(:,iter);Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    Z(:,iter+1)=h_theta(C(:,iter),t);
  end
end