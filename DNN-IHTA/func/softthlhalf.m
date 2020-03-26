% l0.5_th - soft threshold function for L0.5 regularization
function [STq,pt,ss]=softthlhalf(q,t)
pt=abs(q)-t.^(2/3)*0.75*2^(1/3);pt(pt>0)=1;pt(pt<=0)=0;
temp=1+cos(2/3*pi-2/3*acos(t/8.*(abs(q)/3).^(-1.5)));
STq=(2/3)*q.*temp.*pt;
ss=abs(STq);