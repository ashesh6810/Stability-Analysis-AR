clear all;close all;clc

clear all;close all;clc

theta = linspace(-pi,pi,100);
x=cos(theta)+1*i*sin(theta);
load Amatrix_KS_twostep_directstep_11_ICs_lead1.mat;
load Amatrix_KS_Eulerstep_11_ICs_lead1.mat;
load Amatrix_KS_RK4step_11_ICs_lead1.mat;

error = zeros([11,3]);
T=[1,100,500,1000,3000,5000,7000,9000,10000,15000,19000];
figure(1)
set(0, 'DefaultAxesFontSize', 20)

count=1;
for IC = 1:11
% 
[v_direct,eig_direct] = eig(squeeze(A_matrix_direct(IC,:,:)));
[eig_direct, ind] = sort(diag(eig_direct));

[v_Euler,eig_Euler] = eig(squeeze(A_matrix_Euler(IC,:,:)));
[eig_Euler, ind] = sort(diag(eig_Euler));

[v_RK4,eig_RK4] = eig(squeeze(A_matrix_RK4(IC,:,:)));
[eig_RK4, ind] = sort(diag(eig_RK4));



error(1,IC)=(abs(eig_direct(end))-1);
error(2,IC)=(abs(eig_Euler(end))-1);
error(3,IC)=(abs(eig_RK4(end))-1);


end


plot(T,error(1,:),'c*','MarkerFaceColor','c','MarkerSize',20);hold on;
plot(T,error(2,:),'bo','MarkerFaceColor','b','MarkerSize',20);hold on;
plot(T,error(3,:),'ks','MarkerFaceColor','k','MarkerSize',20);hold on;




