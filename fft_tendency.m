clear all;clc;

load('predicted_KS_Eulerstep_lead1.mat');

x=linspace(-50,50,512);
sample=1;
time_step = 19000;
truth =Truth(1:sample:time_step,:);
prediction = prediction(1:time_step,:);

pred_fft = abs(fft(prediction,[],2));
truth_fft = abs(fft(truth,[],2));


truth_dt = truth(2:end,:)-truth(1:end-1,:);
prediction_dt = prediction(2:end,:)-prediction(1:end-1,:);

pred_dt_fft = abs(fft(prediction_dt,[],2));
truth_dt_fft = abs(fft(truth_dt,[],2));



set(0, 'DefaultAxesFontSize', 20)

%loglog([0:255],truth_dt_fft(1000,1:256),'k','Linewidth',2);
loglog([0:255],pred_dt_fft(1,1:256),'b','Linewidth',2)

