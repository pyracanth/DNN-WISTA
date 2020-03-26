function [ dW, dH, dt, dX ] = DNNWISTA_bprop( D, X, Z, Zstar, W, H, t, C, B, T, info)
%WISTA_BPROP
  dC=zeros(size(C));
  dB=zeros(size(B));
  dH=zeros(size(H));
  dZ=zeros(size(Z));
  switch info.netlearn
      case 1
          dZ(:,T+1)=Z(:,T+1)-Zstar;
      case 2
          Z1=Z(:,T+1);Z1(Z1==0)=info.rzero;
          dZ(:,T+1)=D'*(D*Z(:,T+1)-X)+info.lambda*abs(Z1).^(info.p-1).*sign(Z(:,T+1));
  end
  %%
  for layer=T:-1:1
    Z1=Z(:,layer);Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    dC(:,layer)=h_prime(C(:,layer),t)*dZ(:,layer+1);
    dt=-sign(C(:,layer)).*dC(:,layer);
    dB=dB+dC(:,layer);
    dH=dH+dC(:,layer)*Z(:,layer)';
    dZ(:,layer)=H'*dC(:,layer)+ ...
        info.lambda*(info.p-1)*dt.*abs(Z1).^(info.p-2).*sign(Z(:,layer))/info.alpha;
  end
  dB=dB+h_prime(B,t)*dZ(:,1);
  dt=-sign(B).*(h_prime(B,t)*dZ(:,1));
  dW=dB*X';
  dX=W'*dB;
end