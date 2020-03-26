function [Dictionary, data, coefs, borig] = gererateDataB(N, p, dim, K, SNRdB)
%[Aorig, btrain, xorig, borig] = gererateSyntheticDictionaryAndData(numtrain, div, rows, cols,  params.SNRdB);

randn('state',sum(100*clock));
rand('state',sum(100*clock));

Dictionary = randn(dim,K);
%Dictionary = Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
% Normalize columns (Frobenius norm = 1)
for i = 1:K
    nrm = norm(Dictionary(:, i) );
    Dictionary(:, i) = Dictionary(:, i) / (sqrt(K) * nrm);
end

[data,coefs] = CreateDataFromDictionarySimple(Dictionary, N, p);
borig= data;

if (SNRdB==0) | (SNRdB == 80) 
    return
else
    noise = randn(size(data));
    actualNoise = calcNoiseFromSNR(SNRdB,data, noise);
    SNR = calcSNR(data, data+actualNoise);
    data =  data + actualNoise*SNR/SNRdB;   
end

function [D,xOrig] = CreateDataFromDictionarySimple(dictionary, numElements, p)

%randn with min value 1
xOrig = randn(size(dictionary,2),numElements);
xOrig = xOrig;%+0.5*sign(xOrig);

%Bernoulli distribution supply
condition=0;
while condition==0
xsup = (rand(size(dictionary,2),numElements)<=p);
if min(sum(xsup))>0
    condition=1;
end
end
xOrig=xOrig.*xsup;
xOrig = xOrig*diag(1./sqrt(sum(xOrig.*xOrig)));%/sqrt(numElements);  
D = dictionary*xOrig;

function  actualNoise = calcNoiseFromSNR(TargerSNR, signal, randomNoise)
signal = signal(:);
randomNoiseRow = randomNoise(:);
signal_2 = sum(signal.^2);
ActualNoise_2 = signal_2/(10^(TargerSNR/10));
noise_2 = sum(randomNoiseRow.^2);
ratio = ActualNoise_2./noise_2;
actualNoise = randomNoiseRow.*repmat(sqrt(ratio),size(randomNoiseRow,1),1);
actualNoise = reshape(actualNoise,size(randomNoise));

function SNR = calcSNR(origSignal, noisySignal)
errorSignal = origSignal-noisySignal;
signal_2 = sum(origSignal.^2);
noise_2 = sum(errorSignal.^2);

SNRValues = 10*log10(signal_2./noise_2);
SNR = mean(SNRValues);
