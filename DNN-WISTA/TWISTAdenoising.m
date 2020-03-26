clear 
currentpath=cd;
addpath(genpath(currentpath),'-begin');
fig=1;
network.T=10;
network.T1=10;
figno=50:1:60;
info.netlearn=2;% 1for supervised 2for unsupervised
info.checkskip=1;% 1for skip checking
info.Zstarskip=1;% 1for skip generate Zstar for supervised learning
info.p=0.5; %p value for WISTA
sigma = 25; %nosielevel for image
switch info.p
    case 0.9
        info.lambda=57;
    case 0.7
        info.lambda=69;
    case 0.5
        info.lambda=72;
end
info.lj=7;
network.conv_thres=1e-2;
CONV_COUNT=5;
switch info.netlearn
    case 1
        LEARNING_RATE.alpha=9e7;
        LEARNING_RATE.t0=8e9;
    case 2
        LEARNING_RATE.alpha=9e7;%
        LEARNING_RATE.t0=20e9;%
end
LEARNING_RATE.max_change=0.1;
LEARNING_RATE.momentum=0.9;
LEARNING_RATE.LDiffth=1;
LEARNING_RATE.multi=1.5; 
MAX_ITER=5000;
NUM_CLASSES=10;
info.thres=1e-5;
ERROR_CHECK_ITER=10;
% Data generation paramters
info.m = 144;           % Size of A
info.n = 256;
info.rzero=1e-5;
info.non0c = 0.05; internal=2; 
K=256; N1=12;

% Gather the data from image
y0=imread('Lena512.png');
y0=double(y0); % 
%add noise
noise=randn(size(y0));
y1=y0+sigma*noise; % add noise
PSNRinput=10*log10(255^2/mean((y1(:)-y0(:)).^2));

%initialize the dictionary
D=zeros(N1,sqrt(K));
for k=0:1:sqrt(K)-1
    V=cos([0:1:N1-1]*k*pi/sqrt(K));
    if k>0, V=V-mean(V); end
    D(:,k+1)=V/norm(V);
end
D=kron(D,D);
D=D*diag(1./sqrt(sum(D.*D)));
IMdict=Chapter_12_DispDict(D,sqrt(K),sqrt(K),N1,N1,0);
figure(fig); clf; 
imagesc(IMdict); colormap(gray(256)); axis image; axis off; drawnow;
fig=fig+1;
%%  sparse coding
NoTotal=size(y1,1)-N1+1;cnt=1;
Xtrain=zeros(N1^2,((NoTotal-1)/internal+1)^2);
for j=1:internal:NoTotal
    for i=1:internal:NoTotal
        patch=y1(i:i+N1-1,j:j+N1-1);
        Xtrain(:,cnt)=patch(:);
        cnt=cnt+1;
    end
end
network.error=Inf;
info.alpha=max(eig(D'*D))*1.01;
network.W=D'/info.alpha;network.W0=network.W;network.W1=network.W;
network.t=info.lambda/info.alpha*ones(info.n,1);
network.t0=network.t;network.t1=network.t;
network.H=eye(size(D'*D))-1/info.alpha*(D'*D);
network.H0=network.H;network.H1=network.H;
network.iter=0;

[network, Z, Zstar, info]=DNNWista_denoise(D,Xtrain,network, info, NUM_CLASSES,...
    LEARNING_RATE,MAX_ITER,CONV_COUNT,ERROR_CHECK_ITER);

[yout]=RecoverImage(y1,D,Z,internal);
PSNRoutput2=10*log10(255^2/mean((yout(:)-y0(:)).^2));

fprintf('denoising time: %.2fs; network training time: %es\n', info.time, info.trainingtime) ;
disp([PSNRinput, PSNRoutput2]);
figure(figno(1)); clf;
subplot(2,1,1)
imagesc(y1); colormap(gray(256)); axis image; axis off; drawnow;
title(sprintf('Original input PSNR: %.2f dB\n',PSNRinput))
subplot(2,1,2)
imagesc(yout); colormap(gray(256)); axis image; axis off; drawnow;
title(sprintf('TWISTA%.1f output PSNR: %.2f dB\n',info.p,PSNRoutput2))
Record=[info.p;PSNRinput;PSNRoutput2;info.time;info.lambda;LEARNING_RATE.t0;info.lj;LEARNING_RATE.momentum;info.trainingtime;info.rzero];

if info.checkskip==0
figure(figno(2));clf
hold on
plot(info.LX);
if info.Zstarskip==0
plot(info.LZ);
end
set(gca, 'YScale', 'log')
legend('Loss decoder','Loss Zstar');
title('loss func value')
xlabel('Epoch');
end