function  [ ZT, Z1, B ] = Wista_fpropcg( X, W, H, info, T, maxiter )
%WISTA_FPROP
  B(:,:,1)=W*X;
  Z=zeros(info.n,size(X,2),maxiter+2);
  t=info.lambda/info.alpha;
  Z(:,:,2)=softthl1(B,t);
  for iter=1:maxiter
    B(:,:,iter+1)=B(:,:,iter)+H*(Z(:,:,iter+1)-Z(:,:,iter));
     t=info.lambda*abs(Z(:,:,iter+1)).^(info.p-1)/info.alpha;
    Z(:,:,iter+2)=softthl1(B(:,:,iter+1),t);
     if norm(Z(:,:,iter+2)-Z(:,:,iter+1))/norm(Z(:,:,iter+1))<1e-5
        break
    end
  end
  ZT=Z(:,:,T+2);
  Z1=Z(:,:,maxiter+2);
end