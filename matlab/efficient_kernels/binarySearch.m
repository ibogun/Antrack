function [b,c]=binarySearch(x,searchfor)
%% BINARYSEACH
% Code from 
% http://stackoverflow.com/questions/20166847/faster-version-of-find-for-sorted-vectors-matlab
a=1;
b=numel(x);
c=1;
d=numel(x);
while (a+1<b||c+1<d)
    lw=(floor((a+b)/2));
    if (x(lw)<searchfor)
        a=lw;
    else
        b=lw;
    end
    lw=(floor((c+d)/2));
    if (x(lw)<=searchfor)
        c=lw;
    else
        d=lw;
    end
end

% Hack. So that when x(end)< searchfor -> r=end
if (x(b)<searchfor)
    c=c+1;
end
end