function  [ Z, B ] = Wista_fprop( X, W, H, t, info, T )
%WISTA_FPROP unsupervised
  B(:,1)=W*X;
  Z=zeros(size(B,1),T+2);
  Z(:,2)=softthl1(B,t);
  for iter=1:T
    B(:,iter+1)=B(:,iter)+H*(Z(:,iter+1)-Z(:,iter));
    Z1=Z(:,iter+1);Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    Z(:,iter+2)=softthl1(B(:,iter+1),t);
  end
end