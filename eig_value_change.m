clear all;close all;clc

theta = linspace(-pi,pi,100);
x=cos(theta)+1*i*sin(theta);
load Amatrix_KS_one_layer_directstep_11_ICs_lead1.mat;
load Amatrix_KS_one_layer_Eulerstep_11_ICs_lead1.mat;
load Amatrix_KS_one_layer_RK4step_11_ICs_lead1.mat;


% load Amatrix_KS_Eulerstep_20_ICs_lead1.mat ;



vidfile = VideoWriter('Eigenvalues_shallow_changing.mp4','MPEG-4');
vidfile.FrameRate=10;
open(vidfile);

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

plot(x,'r','Linewidth',2);hold on;

% plot(eig_direct(512:-1:502),'co','MarkerSize',10,'MarkerFaceColor','c');
plot(eig_direct,'c*','MarkerSize',10,'MarkerFaceColor','k');
plot(eig_Euler,'ko','MarkerSize',10,'MarkerFaceColor','k');
plot(eig_RK4,'bs','MarkerSize',10,'MarkerFaceColor','k');


% xlim([-1, 1.5]);
legend('Unit Circle','Direct','Euler','RK4');
xlabel('$Re(\lambda)$','Interpreter','latex')
ylabel('$Im(\lambda)$','Interpreter','latex')

pause(0.5);

FF(count)=getframe(gcf);
writeVideo(vidfile, FF(count));
count=count+1;


hold off;

end

close(vidfile);
