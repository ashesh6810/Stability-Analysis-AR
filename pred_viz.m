clear all;clc;

load('predicted_twostep_KS_directstep_lead1.mat');

x=linspace(-50,50,512);
sample=1;
time_step = 19000;
truth =Truth(1:sample:time_step,:);
prediction = prediction(1:time_step,:);
pred_fft = abs(fft(prediction,[],2));

truth_fft = abs(fft(truth,[],2));


for k = 1:size(truth,1)

   u = prediction(k,:);
   v = truth(k,:);

   rmse(k)=norm((u(:)-v(:)),2);

end
   



% set(0, 'DefaultAxesFontSize', 20)
% 
% figure(1)
% 
% loglog([0:255],pred_fft(1,1:256),'-k','Linewidth',2);hold on;
% loglog([0:255],truth_fft(1,1:256),'-r','Linewidth',2);
% 
% legend('RK4 Net','Truth')
% 
% 
% 
% 
% set(0, 'DefaultAxesFontSize', 20)
% 
% figure(2)
% 
% subplot(2,2,1)
% plot(x,prediction(1,:),'b','Linewidth',2);hold on
% plot(x,Truth(1,:),'r','Linewidth',2)
% title(['Time Step' num2str(1)])
% 
% subplot(2,2,2)
% plot(x,prediction(10,:),'b','Linewidth',2);hold on
% plot(x,Truth(10,:),'r','Linewidth',2)
% title(['Time Step' num2str(10)])
% 
% subplot(2,2,3)
% plot(x,prediction(100,:),'b','Linewidth',2);hold on
% plot(x,Truth(100,:),'r','Linewidth',2)
% title(['Time Step' num2str(100)])
% 
% 
% subplot(2,2,4)
% plot(x,prediction(1000,:),'b','Linewidth',2);hold on
% plot(x,Truth(1000,:),'r','Linewidth',2)
% title(['Time Step' num2str(1000)])
% 


figure(3)

plot(rmse(1:19000),'c','Linewidth',2);hold on;









