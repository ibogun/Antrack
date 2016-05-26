
lambda_e=1:3;
lambda_s=0:5;

for i = 1:length(lambda_e)
    for j=1:length(lambda_s)
        s = strcat('ObjStruck_e0_', num2str(lambda_e(i)),'_s0_',num2str(lambda_s(j)));
        saveMatFilesOTB100(s,'OPE');
    end
end