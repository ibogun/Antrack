classdef IntersectionKernel_approx< handle
    %INTERSECTIONKERNEL_APPROX Approximate intersection kernel
    %   Usage
    
    properties
        
        beta;
        X;
        b;
        
        h;
    end
    
    methods
        
        function obj=IntersectionKernel_approx(beta,X,b)
            % beta - coefficients,
            % X    - matrix m x n,
            % b    - m dim vector which is number of points to use for approximation
            
            if(nargin<3)
                b=100;
            end
            
            obj.beta=beta;
            obj.X=X;
            obj.b=b;
            
        end
        
        
        function r=calculateDimension(obj,s,i)
            % function which calculates h_i(s)
            r=dot(obj.beta,min(s,obj.X(:,i)));
        end
        
        
        function r=calculate(obj,x1)
            
            % find min accross one dimension multiply with dot product with
            % beta
            
            n=size(x1,1);
            
            r=zeros(n,1);
            
            for i=1:n
                r(i)=calculateOneDim(obj,x1(i,:));
            end
            
        end
        
        
        function v=calculateOneDim(obj,x)
            v=0;
            [~,n]=size(obj.X);
            for i=1:n
                v=v+calculateDimension(obj,x(i),i);
            end
        end
        
        
        function approximate(obj)
            
            if(~isempty(obj.h))
                return;
            end
            
            [~,n]=size(obj.X);
            
            obj.h=cell(n,1);
            
            for i=1:n
                obj.h{i}=approximateAtIndex(obj,i);
            end
            
        end
        
        
        function [spline, pts, vals] = approximateAtIndex(obj,i)
            
            xi=obj.X(:,i);
            
            minV=min(xi);
            maxV=max(xi);
            
            
            pts=linspace(minV,maxV,obj.b);
            
            vals=zeros(size(pts));
            
            % calculate points value
            for j=1:obj.b
               vals(j)= calculateDimension(obj,pts(j),i);
            end
            
            str=strcat('Intersection kernel interpolant for h_',num2str(i));
            spline=ppcreate(pts,vals,'spline',str);
            
            obj.h{i}=spline;
        end
        
        
    end
    
end

