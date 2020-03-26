function [ Z, C, B ] = dnnihta_fprop( X, W, H, t, T )
%DNNIHTA_FPROP Summary of this function goes here
%   Detailed explanation goes here
  B=W*X;
  C=zeros(numel(B),T);
  Z=zeros(numel(B),T+1);
  Z(:,1)=softthlhalf(B,t);
  for layer=1:T
    C(:,layer)=B+H*Z(:,layer);
    Z(:,layer+1)=softthlhalf(C(:,layer),t);
  end
end