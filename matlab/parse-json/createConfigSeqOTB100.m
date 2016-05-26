
clc; clear; close all;
seqs50=getSequenceNamesOTB50;
seqs100=getSequenceNamesOTB100;

newSeqs=setdiff(lower(seqs100),seqs50);
loadRoot = '/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/results/';

p='''';

dot=',...\n';


trackerName = {'RawStruck'};
%trackerName = {'RobStruck'};
%evalType = {'OPE', 'SRE', 'TRE'};
evalType = {'OPE'};
seqNames = newSeqs;
% for loop -> tracker
trackerID = 1; % len(trackerName)
evalTypeID = 1;
path='/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/';
str=['seqNew={'];

tracker = trackerName{trackerID};
fprintf(str);
% for loop name
for k = 1: length(seqNames)

    filename = strcat(loadRoot,'/' ,evalType{evalTypeID},'/', tracker,...
        '/', seqNames{k},'.json');
    
    fid = fopen(filename,'rt');
    tmp = textscan(fid,'%s','Delimiter','\n', 'BufSize', 12312390);
    fclose(fid);
    
    results = JSON.parse(tmp{1}{1});
    startFrame=results{1}.startFrame;
    endFrame=results{1}.endFrame;
    
    seqName=seqNames{k};
    pre=['struct(',p,'name',p,',',p,seqName,p,',',p,'path',p,',',p,path,seqName,'/',p,',',p,...
        'startFrame',p,',' num2str(startFrame),',', p,'endFrame',p,',',num2str(endFrame),...
        p,',',p,'nz',p,',',num2str(4),',',p,'ext',p,',',p,'jpg',p,',',p,'init_rect',p,',', '[0,0,0,0])'];
    
    if (k== length(seqNames))
        fprintf([pre, '};\n']);
        %str=[str,pre, '};'];
    else
        fprintf([pre,dot,'']);
    end
end