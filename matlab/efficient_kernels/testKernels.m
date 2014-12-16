n=300;
m=200;

nTestCases=1000;
T=1;
%% Generate two vectors
x1=rand(nTestCases,n)*T;
x2=rand(m,n)*T;

%x1=[1 2 3];
%x2=[2 4 6; 2 1 0];

beta=rand(m,1)*T;         % vector of coeficients

beta(1)=0;
beta(1)=-sum(beta);

t_regular=0;
t_fast=0;

tic;
z=IntersectionKernel(beta,x2);
t_regular=t_regular+toc;

tic;
x=IntersectionKernel_fast(beta,x2);
t_fast=t_fast+toc;

fprintf(' Preprocessing regular %f \n',t_regular);
fprintf(' Preprocessing fast %f \n',t_fast);

for i=1:nTestCases
    tic;
    r1=z.calculate(x1(i,:));
    t_regular=t_regular+toc;
    
    tic;
    r2=x.calculate(x1(i,:));
    t_fast=t_fast+toc;
    
    assert(abs(r1-r2)<1e-6,' Kernels should be equal');
end

fprintf(' Time using regular %f \n',t_regular);
fprintf(' Time using fast %f \n',t_fast);