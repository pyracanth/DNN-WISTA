clear 
currentpath=cd;
addpath(genpath(currentpath),'-begin');

network.T=15;
network.T1=network.T;
figno=50:1:60;
info.netlearn=2;% 1for supervised 2for unsupervised
info.p=0.5;
LEARNING_RATE.alpha=0e1;
LEARNING_RATE.t0=5e0;
info.lambda=8e-6;
network.conv_thres=5e-4;
CONV_COUNT=3;
LEARNING_RATE.max_change=0.1;
LEARNING_RATE.momentum=0.9;
LEARNING_RATE.LDiffth=0;
LEARNING_RATE.multi=2; 
MAX_EPOCH=100;
NUM_CLASSES=1;
info.thres=1e-5;
ERROR_CHECK_ITER=1;
% Data generation paramters
info.m = 250;           % Size of A
info.n = 500;
info.rzero=1e-3;
param.p = 0.05;
info.non0c = 0.05;
num_train=100;
param.SNRdB = 20;
maxtrial=1;
for nt=1:maxtrial
[D, Xtrain, Zorig, Xorig] = gererateDataB(num_train, param.p, info.m, info.n, param.SNRdB);
% difference between Xorig and Xtrain, SNR
Noiseb=Xorig-Xtrain;
SNRdB=10*log10(sum(Xorig.^2)/sum(Noiseb.^2));
%intial hoyer sparsity of x
diversity_hoyer0 = numerosity_hoyer(Zorig);
avgdiversity_hoyer0= mean(diversity_hoyer0);

network.error=Inf;
info.alpha=max(eig(D'*D))*1.01;
network.W=D'/info.alpha;network.W0=network.W;network.W1=network.W;
network.t=info.lambda/info.alpha*ones(size(Zorig,1),1);
network.t0=network.t;network.t1=network.t;
network.H=eye(size(D'*D))-1/info.alpha*(D'*D);
network.H0=network.H;network.H1=network.H;
network.iter=0;

[network, Z, Z0, ZT, info]=DNNihta_trainfull(D,Xtrain,Zorig,network, info, NUM_CLASSES,...
    LEARNING_RATE,MAX_EPOCH,CONV_COUNT,ERROR_CHECK_ITER);
%hoyer sparsity of x
diversity_hoyerZ = numerosity_hoyer(Z);
avgdiversity_hoyer(2)= mean(diversity_hoyerZ);
diversity_hoyerZ0 = numerosity_hoyer(Z0);
avgdiversity_hoyer(1)= mean(diversity_hoyerZ0);
%err
learnt_dataI=D*Z0;
datafitting(1)=0.5*norm(learnt_dataI-Xtrain)^2;
regularization(1)=sum(sum(abs(Z).^info.p));
learnt_data=D*Z;
datafitting(2)=0.5*norm(learnt_data-Xtrain)^2;
regularization(2)=sum(sum(abs(Z).^info.p));
learnt_dataT=D*ZT;
%record
pos0=abs(Zorig)>0;pos1=abs(Z0)>0;pos2=abs(Z0)>info.non0c;pos3=abs(Zorig)>info.non0c;
result.accnt(1,:)=sum(pos0==pos1)/info.n;result.accnt(2,:)=sum(pos3==pos2)/info.n;
pos1=abs(Z)>0;pos2=abs(Z)>info.non0c;
result.accnt(3,:)=sum(pos0==pos1)/info.n;result.accnt(4,:)=sum(pos3==pos2)/info.n;
pos1=abs(ZT)>0;pos2=abs(ZT)>info.non0c;
result.accnt(5,:)=sum(pos0==pos1)/info.n;result.accnt(6,:)=sum(pos3==pos2)/info.n;

result.Zerrnt(1,:)=sum((Z0-Zorig).^2)./sum((Zorig).^2);
result.Zerrnt(2,:)=sum((Z-Zorig).^2)./sum((Zorig).^2);
result.Zerrnt(3,:)=sum((ZT-Zorig).^2)./sum((Zorig).^2);
result.datafitting(1,:)=0.5*sum((D*Z0 - Xtrain).^2);
result.datafitting(2,:)=0.5*sum((D*Z - Xtrain).^2);
result.datafitting(3,:)=0.5*sum((D*ZT - Xtrain).^2);
end
result.Zerrsum(1,:)=mean(result.Zerrnt,2)';
result.Zerrsum(2,:)=std(result.Zerrnt,0,2)';
result.datafittingsum(1,:)=mean(result.datafitting,2)';
result.datafittingsum(2,:)=std(result.datafitting,0,2)';
result.acc(1,:)=mean(result.accnt,2)'*100;
result.acc(2,:)=std(result.accnt,0,2)'*100;
Record=[result.acc(:,1),result.acc(:,3),result.acc(:,5); ...
    result.acc(:,2),result.acc(:,4),result.acc(:,6); ... 
    result.Zerrsum;result.datafittingsum];
Record1=[info.m,info.n;info.lambda,param.p;num_train,info.trainingtime;LEARNING_RATE.alpha,LEARNING_RATE.t0;param.SNRdB, info.rzero];

%plot
figure(figno(1));clf
subplot(2, 1, 1);
hold on
plot(info.Zerr,'blue');
plot(info.Z1err,'black');
title('Z error');
hold off
subplot(2, 1, 2);
plot(info.avgdiversity_hoyer);
hold on
plot(info.avgdiversity_hoyer1,'black');
plot(info.avgdiversity_hoyer0(ones(1,info.convergeiter)),'.red');
hold off
axis([0 info.convergeiter 0 1]);
title(sprintf('Original Hoyer sparseness = %.3g , Average Hoyer sparseness = %.3g', info.avgdiversity_hoyer0, info.avgdiversity_hoyer1(network.T1)) );
xlabel('Iteration');
ylabel('Average Diversity hoyer');
xlabel('Iteration');

figure(figno(2));clf
hold on
plot(info.LZ);
plot(info.LX);
legend('Loss Zstar','Loss decoder');
title('loss func value')
xlabel('Epoch');
