
clc;clear; close all;
% reset random number generator
rng('default');

n=300;
m=200;

nTestCases=10;
T=1;
%% Generate two vectors
x1=rand(nTestCases,n)*T;
x2=rand(m,n)*T;

% x1=[1 2 3];
% x2=[2 4 6; 2 1 0];

beta=rand(m,1)*T-0.5;         % vector of coeficients

beta=beta-sum(beta)/m;
% beta=[2 -2]';


assert(abs(sum(beta))<1e-6,'beta coefficients have to sum to zero');
%% Compute kernel v1.0 (loops)  Complexity: O(mn)

kernel_value_loops=0;
kernel_loops=zeros(1,n);
for i=1:m
    for j=1:n
        
        kernel_value_loops=kernel_value_loops+beta(i)*min(x1(j),x2(i,j));
        kernel_loops(j)=kernel_loops(j)+beta(i)*min(x1(j),x2(i,j));
    end
end



%% Compute kernel v3.0  ( efficient kernel computation) Complexity: O(m log n)

% Step #1: sort everything in increasing order, that is for every i=1...m
% sort elements fr

z=IntersectionKernel(beta,x2);
x=IntersectionKernel_fast(beta,x2);
c=IntersectionKernel_approx(beta,x2,50);

r1=z.calculate(x1);
r2=x.calculate(x1);
r3=c.calculate(x1);

[spline, pts,vals]=c.approximateAtIndex(50);

newpts=rand(1,n);

vals_newpts=spline(newpts);

[~,I]=sort(newpts);

plot(pts,vals,newpts(I),vals_newpts(I));



assert(norm(r1-r2)<1e-6,' values should be equal');
assert(norm(r1-r3)<1e-6,' values should be equal');
% TODO: test if x = x_l for some l -> should be zero always