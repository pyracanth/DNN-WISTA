function [Z, info] = mass_Wistadn_fpropcg10( X, W, H, t, T,info)
%WISTA_FPROP
B=W*X;
Z=softthl1(B,t);
Zk=Z;
for iter=1:T
    Z1=Z;Z1(Z1==0)=info.rzero;
    t=info.lambda*abs(Z1).^(info.p-1)/info.alpha;
    Z=softthl1(B+H*Z,t);
    info.Zchange(iter)=norm(Z-Zk)/norm(Zk);
    if mod(iter,10)==1
        fprintf('Iteration: %d Changing rate: %d \n', iter, info.Zchange(iter)) ;
    end
    if info.Zchange(iter)<info.tol
        info.convergeiter=iter;
        break
    end
    Zk=Z;
end
fprintf('Iteration: %d Changing rate: %d \n', iter, info.Zchange(iter))
end