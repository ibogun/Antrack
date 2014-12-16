
clc;clear; close all;
% reset random number generator
%rng('default');

n=300;
m=200;

T=100;
%% Generate two vectors
x1=rand(1,n)*T;
x2=rand(m,n)*T;

% x1=[1 2 3];
% x2=[2 4 6; 2 1 0];

beta=rand(m,1)*T;         % vector of coeficients

beta(1)=0;
beta(1)=-sum(beta);
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

%% Compute kernel v2.0 ( vectorized code )   Complexity: O(mn)
minFunc = @(x1,x2) sum(bsxfun(@min,x1,x2),2);
intersectionKernel=@(beta,x1,x2) dot(beta,minFunc(x1,x2));

kernel_value_vectorized=intersectionKernel(beta,x1,x2);


% make sure values are the same
assert(abs(kernel_value_loops-kernel_value_vectorized)<=1e-5);

%% Compute kernel v3.0  ( efficient kernel computation) Complexity: O(m log n)

% Step #1: sort everything in increasing order, that is for every i=1...m
% sort elements from j=1,...,n

[x_s,I]=sort(x2,1,'ascend');

% copy n times
Y=repmat(beta,[1,n]);
% make it in sorted order
Y=Y(I);

% Step #2: precompute matrices A,B which ar defined by
% A_i(r)=sum_{1<=l<=r} beta_l

% [~,r]=binarySearch(x_s(:,j),s);


% do it with loops first

A=zeros(m,n);
B=zeros(m,n);


for i=1:n
    A(1,i)=Y(1,i)*x_s(1,i);
    for r=2:m
        A(r,i)=A(r-1,i)+Y(r,i)*x_s(r,i);
    end
    
    for r=m-1:-1:1
        B(r,i)=B(r+1,i)+Y(r+1,i);
    end
end


h=zeros(m,n);

for i=1:n
    for r=1:m
        
        h(r,i)=A(r,i)+x_s(r,i)*B(r,i);
    end
end

r=1;

% THIS CAN BE USED FOR TESTING
for i=1:n
    assert(abs(A(r,i)+x_s(r+1,i)*B(r,i)-A(r+1,i)+x_s(r+1,i)*B(r+1,i))<1e-6);
end



% A=cumsum(x_s.*Y);   <= correct

k=0;

for i=1:n
    % given s, find largest r: x_s_{r,i}<=s
    v=0;
    [~,r]=binarySearch(x_s(:,i),x1(i));
    
    if (r~=1)
        assert(x_s(r,i)<=x1(i));
    end
    if (x1(i)<x_s(1,i))
        v=0;
    elseif(x1(i)>=x_s(m,i))
        v=h(r,i);
        
    else
%         low=A(r,i)+x_s(r,i)*B(r,i);
%         high=A(r,i)+x_s(r+1,i)*B(r,i);
        
        low=h(r,i);
        high=h(r+1,i);
        % find point on the line from low-high x_s(r,i) - x_s(r+1,i)
        %x_3=x1(i);
        v=((high-low)/(x_s(r+1,i)-x_s(r,i)))*(x1(i)-x_s(r,i))+low;
    end
    
    
    k=k+v;
    
    
    
    
end

%fprintf('--------\n%d \n',abs(k-kernel_value_loops));
fprintf('%d \n',abs(k-kernel_value_vectorized));
%assert(abs(k-kernel_value_vectorized)<=1e-5);


z=IntersectionKernel(beta,x2);

x=IntersectionKernel_fast(beta,x2);

r1=z.calculate(x1);
r2=x.calculate(x1);

assert(abs(r1-r2)<1e-6,' values should be equal');

% TODO: test if x = x_l for some l -> should be zero always