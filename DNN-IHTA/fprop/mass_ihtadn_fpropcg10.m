function [Z, info] = mass_ihtadn_fpropcg10( X, W, H, t, T, info)
%IHTA_FPROP
B=W*X;
Z=softthlhalf(B,t);Zk=Z;
for iter=1:T
    Z=softthlhalf(B+H*Z,t);
    Zchange= norm(Zk-Z)/norm(Zk);
    if mod(iter,10)==1
        fprintf('Iteration: %d Changing rate: %d \n', iter, Zchange) ;
    end
    if Zchange<info.tol
        info.convergeiter=iter;
        break
    end
    Zk=Z;
end
fprintf('Iteration: %d Changing rate: %d \n', iter, Zchange)
end