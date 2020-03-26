function [ dW, dH, dt, dX ] = dnnihta_bprop(D, X, Z, Zstar, W, H, t, C, B, T, info)
%DNNIHTA_BPROP Summary of this function goes here
%%
dC=zeros(size(C));
%%
dB=zeros(size(B));
dH=zeros(size(H));
dtheta=zeros(size(t));
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
    [~,ptbin,~]=softthlhalf(C(:,layer),t);
    tempacos=acos(t/8.*(abs(C(:,layer))/3).^(-1.5));
    tempsin=sin(2*pi/3-2/3*tempacos);
    tempsqrt=(1-27/64*t.^2./abs(C(:,layer).^3)).^(0.5);
    tempdeltat=-(sqrt(3)*2)^-1*C(:,layer).*abs(C(:,layer)).^(-1.5).*tempsin./tempsqrt;
    tempdeltac=2/3+2/3*cos(2*pi/3-2/3*tempacos)+ ...
        sqrt(3)/4*C(:,layer).*t.*abs(C(:,layer)).^(-2.5).*sign(C(:,layer)).*tempsin./tempsqrt;
    dC(:,layer)=ptbin.*tempdeltac.*dZ(:,layer+1);
    dtheta=dtheta+ptbin.*tempdeltat.*dZ(:,layer+1);
    dB=dB+dC(:,layer);
    dH=dH+dC(:,layer)*Z(:,layer)';
    dZ(:,layer)=H'*dC(:,layer);
end
[~,ptbin,~]=softthlhalf(B,t);
tempacos=acos(t/8.*(abs(B)/3).^(-1.5));
tempsin=sin(2*pi/3-2/3*tempacos);
tempsqrt=(1-27/64*t.^2./abs(B.^3)).^(0.5);
tempdeltat=-(sqrt(3)*2)^-1*C(:,layer).*abs(C(:,layer)).^(-1.5).*tempsin./tempsqrt;
tempdeltac=2/3+2/3*cos(2*pi/3-2/3*tempacos)+ ...
    sqrt(3)/4*B.*t.*abs(B).^(-2.5).*sign(B).*tempsin./tempsqrt;
dB=dB+ptbin.*tempdeltac.*dZ(:,1);
dt=dtheta+ptbin.*tempdeltat.*dZ(:,1);
dW=dB*X';
dX=W'*dB;
end
