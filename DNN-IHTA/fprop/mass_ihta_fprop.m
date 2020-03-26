function Z = mass_ihta_fprop( X, W, H, t, T, info)
%LISTA_FPROP
  B=W*X;
 Z=softthlhalf(B,t);Zk=Z;
  for layer=1:T
    Z=softthlhalf(B+H*Z,t);
    if norm(Zk-Z)/norm(Zk)<info.thres*info.N
        break
    end
    Zk=Z;
  end
end