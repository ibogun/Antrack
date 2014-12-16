classdef IntersectionKernel < handle
    %INTERSECTIONKERNEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        beta;
        X;
        kernelFn;
    end
    
    methods
        
        function obj=IntersectionKernel(beta,X)
           
            obj.beta=beta;
            obj.X=X;
            minFunc = @(x1,x2) sum(bsxfun(@min,x1,x2),2);
            intersectionKernel=@(beta,x1,x2) dot(beta,minFunc(x1,x2));
            
            obj.kernelFn=intersectionKernel;
        end
        
        
        function r=calculate(obj,x1)
            % find min accross one dimension multiply with dot product with
            % beta
            
            n=size(x1,1);
            
            r=zeros(n,1);
            
            for i=1:n
                r(i)=obj.kernelFn(obj.beta,x1(i,:),obj.X);
            end
        end
        
    end
    
end

