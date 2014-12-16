classdef KernelCalculationAnalyzer < handle
    %KERNELCALCULATIONANALYZER Class to analyze kernel calculation
    %   Detailed explanation goes here
    
    properties
        
        approxKernel;
        x_eval;
    end
    
    methods
        
        function obj=KernelCalculationAnalyzer(b,beta,X,x_eval)
            % Generate data if it is not passed
            
            if (nargin<4)
                
                rng('default');
                
                n=300;
                m=200;
                
                nTestCases=200;
                T=1;
                %% Generate two vectors
                x_eval=rand(nTestCases,n)*T;
                X=rand(m,n)*T;
                beta=rand(m,1)*T-0.5;         % vector of coeficients
                
                beta=beta-sum(beta)/m;
            else
                assert(size(beta,1)==size(X,1), 'Number of columns in beta and X should be the same');
            end
            
            
            obj.approxKernel=IntersectionKernel_approx(beta,X,b);
            obj.x_eval=x_eval;
        end
        
        
        
        function [vals_exact, vals_approx, x_t]=approximateAtIndex(obj,i)
            
            % approximate kernels with existing b
            % calculate relative errors
            % plot comparison plots
            % output max and averate relative errors
            
            obj.approxKernel.approximate();

            x_t=obj.x_eval(:,i);
            [x_t]=sort(x_t);
            
            % calculate approximated values
            vals_approx=obj.approxKernel.h{i}(x_t);
            
            
            K=length(vals_approx);
            vals_exact=zeros(K,1);
            
            for j=1:K
                vals_exact(j)=obj.approxKernel.calculateDimension(x_t(j),i);
            end
 
        end
        
        
        function rmse=approximate(obj)
            
            [~,n]=size(obj.x_eval);
            
            rmse=zeros(n,1);
            for i=1:n
               [v_exact, v_approx, ~]=approximateAtIndex(obj,i);
               
               % delete zeros -> from v_exact
               badIdx=(v_exact==0);
               v_exact(badIdx)=[];
               v_approx(badIdx)=[];
               rmse(i)=sqrt(sum((v_exact(:)-v_approx(:)).^2)/numel(v_exact));
            end
            
        end
        
    end
    
    methods(Abstract)
        
    end
    
end

