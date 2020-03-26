function [network, Z, info] = TWista_trainVdenoising(D, X, network, num_of_classes, learning_rate, max_iter, error_check_iter, info )
%LWISTA TRAIN for denoising
  %initialize variables
  if info.traindisp==1
      disp(strcat({'Lambda is '}, num2str(info.lambda)));
      disp(strcat({'Network depth is '}, num2str(network.T)));
      disp(strcat({'Convergence threshold is '}, num2str(network.conv_thres)));
  end
  %%
  info.N=size(X,2);
  T=network.T;info.trainingtime=0;
  conv_thres=network.conv_thres;
  best_network=network;
  %%
  conv_count=0;
  %%
%   diversity_hoyer0 = numerosity_hoyer(Zstar);
%   info.avgdiversity_hoyer0= mean(diversity_hoyer0);
  %if isinf(network.error)
      %tic;
      Z=mass_Wista_fprop(X,network.W,network.H,network.t, T,info);
      %toc;
      %err=Zstar-Z;
      %LW1=0.5*mean(sum((err).^2));
      LW1=0.5*norm(D*Z-X,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));      
      skip_first_error_check=false;
      j=0;
%   else
%       LW1=network.error;
%       skip_first_error_check=true;
%       j=network.iter;
%   end
  fprintf('Starting error is: %.2e\n',LW1);
  VW=0; VH=0; Vt=0; 
  while j<max_iter
    %fprintf('Iteration %d\n',j);
    %% WISTA FP and BP
    j=j+1;
    tic;
    idx=ceil(rand*info.N);%mod((j-1)*info.lj,numtrain)+1;
    switch network.train %LWista_trainTnet LWista_trainLnet
        case 'Lnet'
            [Z,C,B]=Wista_fpropLnet(X(:,idx),network.W,network.H,network.t, T, info);
            [dW,dH,dt,~]=Wista_bpropLnet(D,X(:,idx),Z,network.W,network.H,network.t,C,B,T,info);
        case 'Tnet'
            [Z,B]=Wista_fprop(X(:,idx),network.W,network.H,network.t, info,T);
            [dW,dH,dt,~]=Wista_bpropTnet(D,X(:,idx),Z,network.W,network.H,network.t,B,T,info);
    end
    %%
    conv_coef=1/(learning_rate.alpha*...
      (double((idivide(uint64(j-1),uint64(num_of_classes))+1)))+learning_rate.t0);
    VW=learning_rate.momentum*VW+conv_coef*dW;
    VH=learning_rate.momentum*VH+conv_coef*dH;
    Vt=learning_rate.momentum*Vt+conv_coef*dt;
    learning_rate.momentum=min(0.9,learning_rate.momentum+0.005);
    network.W=network.W-VW; %network.W=col_norm(network.W',2)';
    network.H=network.H-VH;
    network.t=network.t-Vt;
    ttime=toc;
    %%
    if (mod(j,error_check_iter)==min([1;error_check_iter-1]) || j==max_iter)...
       && ~skip_first_error_check
      %tic;
      %% LISTA ReFP
      %Z=lista_Rfprop(X,network,VW,VH,Vt);
      Z=mass_Wista_fprop(X,network.W,network.H,network.t,T,info);
      %toc;
%       err=Zstar-Z;
%       LW=0.5*mean(sum((err).^2));
      LW=0.5*norm(D*Z-X,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));
      LDiff=100*(LW-LW1)/LW1;
      if info.traindisp==1
      fprintf('Iteration %d:\n',j);
      mdW=max(abs(VW(:)));
      mdH=max(abs(VH(:)));
      mdt=max(abs(Vt(:)));
      fprintf('dW:    %e\n',mdW);
      fprintf('dH:     %e\n',mdH);
      fprintf('dt: %e\n',mdt);
      %fprintf('mL(W):  %e\n',max(mean(abs(err),1)));
      fprintf('L(W):   %e\n',LW);
      fprintf('L Diff: %f %%\n',LDiff);
      end
      network.error=LW;
      % re
      if j>2 && (LDiff>learning_rate.LDiffth || isnan(LDiff))
          network.W=network.W1; 
          network.H=network.H1;
          network.t=network.t1;  
          learning_rate.t0=learning_rate.t0*learning_rate.multi;
          LW=LW1;        
      else
          network.W1=network.W; 
          network.H1=network.H;
          network.t1=network.t;
          info.trainingtime=info.trainingtime+ttime;
      end
      %%
      if network.error<best_network.error
        best_network=network;
        best_network.iter=j;
      end
      if (abs(LDiff)/100>conv_thres)
        conv_count=0;
      else
        conv_count=conv_count+1;
      end
      if (conv_count==network.conv_count)
        break;
      end
      LW1=LW;
    end
    skip_first_error_check=false;
  end
  
  if isinf(best_network.error); best_network=network; end
  network=best_network;
  tic
  Z=mass_Wista_fprop(X,network.W,network.H,network.t, network.T,info);
  info.time=toc;
%   err=Zstar-Z;
%   LW=0.5*mean(sum((err).^2));
  LW=0.5*norm(D*Z-X,'fro')^2+info.lambda*sum(sum(abs(Z).^info.p));
  fprintf('Best L(W): %.2e  training time: %.2e s; denoising time: %.2e s\n',LW,info.trainingtime,info.time);%disp('Finished');
end