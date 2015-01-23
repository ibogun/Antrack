clc;clear; close all;

display('Dimension of the feature vector is 480');
fast=load('data/fast.mat','-ascii');
regular=load('data/regular.mat','-ascii');
preprocessing_fast=load('data/preprocessing_fast.mat','-ascii');
numLocations=load('data/numLocations.mat','-ascii');
numSupportVectors=load('data/numSupportVectors.mat','-ascii');

approx=load('data/approx_kernel.mat','-ascii');

% for each set of support vectors plot
% time vs numLocationsProcessed

i=1;

lineWidth=2.5;



figure;

for i=1:4
    subplot(2,2,i);
    plot(numLocations,fast(i,:),numLocations,regular(i,:),numLocations,approx(i,:),'LineWidth',lineWidth);
    hold on;
    %linear(numLocations,linear(i,:),'--');
    t=['Number of support vectors ',num2str(numSupportVectors(i))];
    title(t);
    xlabel('# of locations');
    ylabel('Time (seconds)');
    
    legend('fast','regular','approx/50','Location','NorthWest');
end

set(findall(gcf,'type','text'),'FontSize',13,'fontWeight','bold')
set(gcf,'Position',[200 5 900 924]);

saveToFolder='/Users/Ivan/Documents/Papers/Reports/Antrack/images/';
filename=strcat(saveToFolder,'time_efficiency_experiment1.0.pdf');
display(strcat('Results will be saved to ',filename));
save2pdf(filename,gcf,600);