clear all;clc;

load('predicted_KS_RK4step_lead1.mat');

x=linspace(-50,50,512);
sample=1;
time_step = 19000;
truth =Truth(1:sample:time_step,:);
prediction = prediction(1:time_step,:);
pred_fft = abs(fft(prediction,[],2));

truth_fft = abs(fft(truth,[],2));



set(0, 'DefaultAxesFontSize', 20)
colors_pred={'--k','--m'};


% figure(1)
% loglog([0:255],truth_fft(1,1:256),'r','Linewidth',2);hold on;
% 
% count=1;
% 
% for k = 1:size(colors_pred,2)
% 
% loglog([0:255],pred_fft(count,1:256),colors_pred{k},'Linewidth',2);
% 
% count=count+5000;
% 
% end

count=1;
for k = 1:1:19000

    u=pred_fft (k,1:10);
    v=truth_fft(1,1:10);

    mse(count)=sqrt(sum((u(:)-v(:)).^2));

    count=count+1;

end

plot([1:1:19000],mse,'ks','MarkerFaceColor','k');hold on;


