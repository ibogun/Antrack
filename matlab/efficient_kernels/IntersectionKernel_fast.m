classdef IntersectionKernel_fast < handle
    %INTERSECTIONKERNEL_FAST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        beta;
        X;
        
        A;
        B;
        x_s;
        h;
    end
    
    methods
        
        function obj=IntersectionKernel_fast(beta,X)
            
            obj.X=X;
            [m,n]=size(X);
            [obj.x_s,I]=sort(X,1,'ascend');
            
            % copy n times
            Y=repmat(beta,[1,n]);
            % make it in sorted order
            Y=Y(I);
            
            % Step #2: precompute matrices A,B which ar defined by
            % A_i(r)=sum_{1<=l<=r} beta_l
            
            % [~,r]=binarySearch(x_s(:,j),s);
            
            
            % do it with loops first
            
            obj.A=zeros(m,n);
            obj.B=zeros(m,n);
            
            
            for i=1:n
                obj.A(1,i)=Y(1,i)*obj.x_s(1,i);
                for r=2:m
                    obj.A(r,i)=obj.A(r-1,i)+Y(r,i)*obj.x_s(r,i);
                end
                
                for r=m-1:-1:1
                    obj.B(r,i)=obj.B(r+1,i)+Y(r+1,i);
                end
            end
            
            
            obj.h=zeros(m,n);
            
            for i=1:n
                for r=1:m
                    
                    obj.h(r,i)=obj.A(r,i)+obj.x_s(r,i)*obj.B(r,i);
                end
            end
            
        end
        
        
        function k=calculate(obj,x1)
            
            [m,n]=size(obj.X);
            k=0;
            
            for i=1:n
                % given s, find largest r: x_s_{r,i}<=s
                v=0;
                [~,r]=binarySearch(obj.x_s(:,i),x1(i));
                
                if (x1(i)<obj.x_s(1,i))
                    v=0;
                elseif(x1(i)>=obj.x_s(m,i))
                    v=obj.h(r,i);
                    
                else
                    %         low=A(r,i)+x_s(r,i)*B(r,i);
                    %         high=A(r,i)+x_s(r+1,i)*B(r,i);
                    
                    low=obj.h(r,i);
                    high=obj.h(r+1,i);
                    % find point on the line from low-high x_s(r,i) - x_s(r+1,i)
                    %x_3=x1(i);
                    v=((high-low)/(obj.x_s(r+1,i)-obj.x_s(r,i)))*(x1(i)-obj.x_s(r,i))+low;
                end
                
                
                k=k+v;
                
                
                
                
            end
        end
    end
    
end

