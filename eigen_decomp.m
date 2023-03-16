clear all;clc

theta = linspace(-pi,pi,100);
x=cos(theta)+1*i*sin(theta);

IC = 4;



load Amatrix_KS_directstep_11_ICs_lead1.mat;
load Non_sense_Amatrix_KS_Euler_lead1.mat;
load Amatrix_KS_RK4step_4ICs_lead1.mat;

[v_euler,eig_Euler] = eig(squeeze(A_matrix_Euler));
%[v_euler,eig_Euler] = eig(squeeze(A_matrix_Euler(IC,:,:)));

[eig_Euler, ind] = sort(diag(eig_Euler));


[v_direct,eig_direct] = eig(squeeze(A_matrix_direct(IC,:,:)));
[eig_direct, ind] = sort(diag(eig_direct));

[v_RK4,eig_RK4] = eig(squeeze(A_matrix_RK4(IC,:,:)));
[eig_RK4, ind] = sort(diag(eig_RK4));

figure(1)
set(0, 'DefaultAxesFontSize', 20)

subplot(2,2,IC) 
plot(x,'r','Linewidth',2);hold on;

plot(eig_Euler,'ks', 'MarkerSize',10, 'MarkerFaceColor','k');hold on;
plot(eig_direct,'co','MarkerSize',10,'MarkerFaceColor','c');
plot(eig_RK4,'ro','MarkerSize',10,'MarkerFaceColor','r');

legend('Unit Circle','Euler','Direct','RK4')
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')





%plot(diag(eig_Euler),'k*')
