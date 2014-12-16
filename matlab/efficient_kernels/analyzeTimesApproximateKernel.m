
clc;clear;close all;
b=4:100;


saveToFolder='/Users/Ivan/Documents/Papers/Reports/Antrack/images/';
display(strcat('Results will be saved to ',saveToFolder));
%% Plot #1: accuracy of the approximate kernel

index=53;
obj=KernelCalculationAnalyzer(10);
[vals_exact, vals_approx, x_t]=approximateAtIndex(obj,index);



lineWidth=2.6;
h=figure;
plot(x_t,vals_exact,x_t,vals_approx,'LineWidth',lineWidth);
legend('Exact','Approximate');

t=[' Kernel approximation accuracy'];
title(t);
xlabel('s');
ylabel('Kernel value');


set(findall(gcf,'type','text'),'FontSize',13,'fontWeight','bold')

set(gca,'fontsize',13);

save2pdf(strcat(saveToFolder,'approximate_kernel_accuracy.pdf'),h,600);

%% Plot #2: errors vs b
if (exist('rmse_approximateKernel.mat','file')==0)
    
    display('This might take a while... please wait...');
    meanRMSE=zeros(length(b),1);
    for i=1:length(b)
        
        display(i);
        obj=KernelCalculationAnalyzer(b(i));
        
        errs=obj.approximate;
        meanRMSE(i)=mean(errs);
    end
    
    
    save('rmse_approximateKernel','meanRMSE');
    
else
    load('rmse_approximateKernel');
    
end


h1=figure;
plot(b,meanRMSE,'LineWidth',lineWidth);

t=['Error vs #b'];
title(t);
xlabel('b');
ylabel('RMSE');


set(findall(gcf,'type','text'),'FontSize',13,'fontWeight','bold')

set(gca,'fontsize',13);

save2pdf(strcat(saveToFolder,'error_vs_b.pdf'),h1,600);
