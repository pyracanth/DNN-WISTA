function [network, Z, Zstar, ZT, info] = DNNWista_trainfull(D, X, Zorig, network, info, num_of_classes, learning_rate, max_epoch, conv_count_thres, error_check_iter )
%TRAIN Summary of this function goes here
% X: training input signal nxm (m input of size n)
% W: dictionary nxk (k basis vector size n)
% Zstar: kxm (m sparse code with coeffs size k)
% alpha: sparse penalty
% T: depth of the neural network
% P: number of training iteration
%initialize variables
disp(strcat({'Lambda is '}, num2str(info.lambda)));
disp(strcat({'Network depth is '}, num2str(network.T)));
disp(strcat({'Convergence threshold is '}, num2str(network.conv_thres)));
%%
disnan=1; info.trainingtime=0;
best_network=network;
info.N=size(X,2);
learning_rate.t=learning_rate.t0;
%%
conv_count=0;
%%
diversity_hoyer0 = numerosity_hoyer(Zorig);
info.avgdiversity_hoyer0= mean(diversity_hoyer0);
tic;
[ZT,Zstar,info.convergeiter,info.Zerr,info.avgdiversity_hoyer]= ...
    mass_Wista_errsparsity(Zorig,X,network.W,network.H,network.t,info,network.T, 500);
toc;
if isinf(network.error)
    Z=mass_Wista_fprop(X,network.W,network.H,network.t, network.T,info);
    resid=D*Z-X;
    info.LX(1)=0.5*norm(resid,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));
    err=Zstar-Z;
    info.LZ(1)=0.5*mean(sum((err).^2));
    j=0;
else
    info.LX(1)=network.error;
    j=network.iter;
end
switch info.netlearn
    case 1
        fprintf('Starting error is: %.2e, converge iteration: %d\n',info.LZ(1),info.convergeiter);
    case 2
        fprintf('Starting error is: %.2e, converge iteration: %d\n',info.LX(1),info.convergeiter);
end
VW=0; VH=0; Vt=0; count=2;
while j<max_epoch
    %fprintf('Iteration %d\n',j);
    %% LISTA FP and BP
    tic
    j=j+1;
    for idx=1:1:info.N
        [Z,C,B]=DNNWISTA_fprop(X(:,idx),network.W,network.H,network.t, network.T, info);
        [dW,dH,dt,~]=DNNWISTA_bprop(D,X(:,idx),Z,Zstar(:,idx),network.W,network.H,network.t,C,B,network.T,info);
        %%
        conv_coef=1/(learning_rate.alpha*...
            (double((idivide(uint64(j-1),uint64(num_of_classes))+1)))+learning_rate.t);
        nancd=max([max(max(isnan(dW))),max(max(isnan(dH))),max(max(isnan(dt)))]);
        if nancd
            if disnan
                disp('NAN update occurs');
                disnan=0;
            end
            VW=0; VH=0; Vt=0;
            break;
        else
            VW=learning_rate.momentum*VW+conv_coef*dW;
            VH=learning_rate.momentum*VH+conv_coef*dH;
            Vt=learning_rate.momentum*Vt+conv_coef*dt;
            learning_rate.momentum=min(0.9,learning_rate.momentum+0.005);
            network.W=network.W-VW; %network.W=col_norm(network.W',2)';
            network.H=network.H-VH;
            network.t=network.t-Vt;
        end
    end
    fprintf('Epoch %d:\n',j);
    ttime=toc;
    fprintf('Training time: %.2f seconds\n',ttime);
    %%
    if (mod(j,error_check_iter)==min([1;error_check_iter-1]) || j==max_epoch)
        Z=mass_Wista_fprop(X,network.W,network.H,network.t, network.T1,info);
        resid=D*Z-X;
        info.LX(count)=0.5*norm(resid,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));
        err=Zstar-Z;
        info.LZ(count)=0.5*mean(sum((err).^2));
        LDiff=100*(info.LX(count)-info.LX(count-1))/info.LX(count-1);
        LDiff1=100*(info.LZ(count)-info.LZ(count-1))/info.LZ(count-1);
        fprintf('Learning rate:   %.2e\n',conv_coef);
        fprintf('Learning rate.t:   %.2e\n',learning_rate.t);
        fprintf('L(X):   %.2e\n',info.LX(count));
        fprintf('L Diff: %.2f %%\n',LDiff);
        fprintf('L1(Z):   %.2e\n',info.LZ(count));
        fprintf('L1 Diff: %.2f %%\n',LDiff1);
        disnan=1;
        switch info.netlearn
            case 1                
                network.error=info.LZ(count);
                convdiff=abs(info.LZ(count)-info.LZ(count-1))/info.LZ(count-1);
                LDiffo=LDiff1;
            case 2
                network.error=info.LX(count);
                convdiff=abs(info.LX(count)-info.LX(count-1))/info.LX(count-1);
                LDiffo=LDiff;
        end
        % re
        if (LDiffo>learning_rate.LDiffth || isnan(LDiffo))
            network.W=network.W1;
            network.H=network.H1;
            network.t=network.t1;
            learning_rate.t=learning_rate.t*learning_rate.multi;
            info.LX(count)=info.LX(count-1);
            info.LZ(count)=info.LZ(count-1);
        else
            network.W1=network.W;
            network.H1=network.H;
            network.t1=network.t;
            info.trainingtime=info.trainingtime+ttime;
            if mod(j,5)==0 && (learning_rate.t>learning_rate.t0*3e0)
                learning_rate.t=learning_rate.t/2;
            end
        end        
        count=count+1;
        %%
        if network.error<best_network.error
            best_network=network;
            best_network.iter=j;
        end
        if convdiff>network.conv_thres || isnan(convdiff)
            conv_count=0;
        else
            conv_count=conv_count+1;
        end
        if (conv_count==conv_count_thres)
            break;
        end
    end
end

if isinf(best_network.error); best_network=network; end
network=best_network;
[~,Z,info.totallayer,info.Z1err,info.avgdiversity_hoyer1]=...
    mass_Wista_errsparsity(Zorig,X,network.W,network.H,network.t,info,network.T,network.T1);
switch info.netlearn
    case 1
        err=Zstar-Z;
        Lf=0.5*mean(sum((err).^2));
    case 2
        resid=D*Z-X;
        Lf=0.5*norm(resid,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));
end
fprintf('\nBest L(W):   %.2e\n',Lf);
disp('Finished');
end