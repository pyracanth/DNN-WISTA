clear
currentpath=cd;
addpath(genpath(currentpath),'-begin');
videopath='./Videos/';
info.traindisp=0; %training change display in workspace
info.framedisp=1; %denoising frame display
casenumber=ceil(rand()*3); %for p= 0.9; 0.7; 0.5;
%% Take a random video
videoList=dir([videopath '*.webm']); randnum=ceil(rand()*size(videoList,1));
[~,videoName]=fileparts(videoList(randnum).name);
% read video
Obj = VideoReader([videopath videoName '.webm']);
nFrames = Obj.NumberOfFrames;
%% Initialize
K=256; N1=12; Kdata=12;
frameheight=360;framewidth=480;
network.T=4;
sigma=25; % noise power 
figno=1:1:50;
MAX_ITER=5000;
info.lj=2;
switch casenumber
    case 1
        info.p=0.9;
        info.lambda=45;
    case 2
        info.p=0.7;
        info.lambda=65;
    case 3
        info.p=0.5;
        info.lambda=77;
end
fprintf(['Denoising video <',videoName,'> with WISTA%.1f \n'],info.p);
info.rzero=1e-5;
NUM_CLASSES=10;
network.train='Lnet';
network.conv_thres=1e-2;
network.conv_count=3;
ERROR_CHECK_ITER=10;
Timethre=10e-5;
%initialize the dictionary
Dictionary=zeros(N1,sqrt(K));
for k=0:1:sqrt(K)-1 
    V=cos([0:1:N1-1]*k*pi/sqrt(K));
    if k>0, V=V-mean(V); end
    Dictionary(:,k+1)=V/norm(V);
end
Dictionary=kron(Dictionary,Dictionary);
Dictionary=Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
[info.m,info.n]=size(Dictionary);           % Size of Dic
network.error=Inf;
info.alpha=max(eig(Dictionary'*Dictionary))*1.01;
network.W=Dictionary'/info.alpha;network.We0=Dictionary'/info.alpha;
network.t=info.lambda/info.alpha*ones(K,1);network.t0=info.lambda/info.alpha*ones(K,1);
network.H=eye(size(Dictionary'*Dictionary))-1/info.alpha*(Dictionary'*Dictionary);network.H0=eye(size(Dictionary'*Dictionary))-1/info.alpha*(Dictionary'*Dictionary);
network.iter=0;
Record=zeros(9,nFrames);
count=1;num=1;
for k = 1 : nFrames
    %network parameters
    LEARNING_RATE.alpha=8e7;%500100forN1 %80002000for nonnormalisation
    LEARNING_RATE.t0=10e9;%800;
    LEARNING_RATE.max_change=0.1;
    LEARNING_RATE.momentum=0.9;
    LEARNING_RATE.multi=1.5;
    LEARNING_RATE.LDiffth=0;
    im = read(Obj, k);%read k frame
    y0=double(rgb2gray(im)); y0=y0(1:frameheight,1:framewidth);
    noise=sigma*randn(frameheight,framewidth);
    y1(:,:,k)=y0+noise; % add noise
    y11=y1(:,:,k);
    PSNRinput(k)=10*log10(255^2/mean((y11(:)-y0(:)).^2));
    %convert image into patches
    NoTotal=size(y11,1)-N1+1;cnt=1;
    Ntotal=size(y11,2)-N1+1;
    Data=zeros(N1^2,((NoTotal-1)/Kdata+1)*((Ntotal-1)/Kdata+1));
    Data0=zeros(N1^2,((NoTotal-1)/Kdata+1)*((Ntotal-1)/Kdata+1));
    for j=1:Kdata:Ntotal
        for i=1:Kdata:NoTotal
            patch=y11(i:i+N1-1,j:j+N1-1);
            Data(:,cnt)=patch(:);
            patch=y0(i:i+N1-1,j:j+N1-1);
            Data0(:,cnt)=patch(:);
            %DataZ(:,:,cnt)=patch(:,:);
            cnt=cnt+1;
        end
    end
    %sort data differences
    if k==1%mod(k,50)==1%k==1%
        Data1=Data;
    else
        DataDiff=sum((Data0-Data2).^2)./sum((Data2).^2);
        num=sum(DataDiff>Timethre);
        [valueD,posD]=sort(DataDiff,'descend');
        if num>0
        Data1=zeros(N1^2,num);
        for j=1:num
            Data1(:,j)=Data(:,posD(j));
        end
        end
    end
    Data2=Data0;
    if num>0
        [network, Z1, info]=TWista_trainVdenoising(Dictionary,Data1,network,NUM_CLASSES,...
            LEARNING_RATE,MAX_ITER,ERROR_CHECK_ITER,info);
    else
        info.time=0;
        info.trainingtime=0;
    end
    if k==1
        Z=Z1;
    else
        for j=1:num
            Z(:,posD(j))=Z1(:,j);
        end
    end
[yout(:,:,k)]=RecoverImage(y11,Dictionary,Z,Kdata);
yout1=yout(:,:,k);
PSNRoutput2(k)=10*log10(255^2/mean((yout1(:)-y0(:)).^2));
if info.framedisp==1
figure(figno(1)); clf;
subplot(2,1,1)
imagesc(y11); colormap(gray(256)); axis image; axis off; drawnow;
subplot(2,1,2)
imagesc(yout1); colormap(gray(256)); axis image; axis off; drawnow;
end
fprintf('Frame %d, input PSNR: %.2f dB; Output PSNR: %.2f dB \n',k,PSNRinput(k),PSNRoutput2(k));
end