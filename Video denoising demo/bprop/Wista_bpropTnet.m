function [ dW, dH, dt1, dX ] = Wista_bpropTnet(D, X, Z, W, H, t, B, T, info)
%TWISTA_BPROP Summary of this function goes here
  %%
  Z1=Z(:,T+2);Z1(Z1==0)=info.rzero;
  dZ(:,T+2)=D'*(D*Z(:,T+2)-X)+info.lambda*abs(Z1).^(info.p-1).*sign(Z(:,T+2));
  Z1=Z(:,T+1);Z1(Z1==0)=info.rzero;
  t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
  [~,ptbin,~]=softthl1(B(:,T+1),t);
  dt(:,T+1)=dZ(:,T+2).*sign(B(:,T+1)).*ptbin*(-1);
  dB(:,:,T+1)=dZ(:,T+2).*ptbin;
  dH=zeros(size(H));
  Z1=Z(:,T+1);Z1(Z1==0)=info.rzero;
  dZ(:,T+1)=H'*dB(:,T+1)+info.lambda*(info.p-1)*dt(:,T+1).*abs(Z1).^(info.p-2).*sign(Z(:,T+1))/info.alpha;
   %%
  for layer=T:-1:1
      Z1=Z(:,layer);Z1(Z1==0)=info.rzero;
      if layer ==1
          t=info.lambda/info.alpha;
      else
          t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
      end
      dH=dH+dB(:,:,layer+1)*(Z(:,layer+1)-Z(:,layer))';
      [~,ptbin,~]=softthl1(B(:,layer),t);
      dt(:,layer)=(-1)*ptbin.*sign(B(:,layer)).*(dZ(:,layer+1));
      dB(:,:,layer)=dB(:,:,layer+1)+ptbin.*dZ(:,layer+1);
      dZ(:,layer)=H'*dB(:,:,layer)-H'*dB(:,:,layer+1)+ ...
          info.lambda*(info.p-1)*dt(:,layer).*abs(Z1).^(info.p-2).*sign(Z(:,layer))/info.alpha;
  end
  dW=dB(:,:,1)*X';
  dX=W'*dB(:,:,1);
  dt1=dt(:,1);
end
