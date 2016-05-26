function [  ] = saveMatFilesOTB50( tracker_name, eval_type )
%SAVEMATFILESOTB100 Summary of this function goes here
%   Detailed explanation goes here



saveRoot = '/Users/Ivan/Code/Libraries/matlab_libraries/tracker_benchmark_v1.0/results/';
loadRoot = '/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/results/';

conference = 'results';
trackerName = {tracker_name};
%trackerName = {'RobStruck'};
%evalType = {'OPE', 'SRE', 'TRE'};
evalType = {eval_type};
names = getSequenceNamesOTB50();
% for loop -> tracker
for trackerID = 1: length(trackerName) % len(trackerName)
    
    display(strcat('Working with the tracker: ', trackerName{trackerID}));
    
    % for loop saveLocations & evalType
    for evalTypeID = 1 : length(evalType) % len(evalType)
        
        display(strcat('Experiment type: ', evalType{evalTypeID}));

        tracker = trackerName{trackerID};
        saveLocation = strcat(saveRoot,  ...
             conference, '_',evalType{evalTypeID});

          % for loop name
        for k = 1: length(names)
            
            display(strcat('Sequence name: ', names{k}));

            filename = strcat(loadRoot,'/' ,evalType{evalTypeID},'/', tracker,...
                   '/', names{k},'.json');
               
            fid = fopen(filename,'rt');
            tmp = textscan(fid,'%s','Delimiter','\n', 'BufSize', 12312390);
            fclose(fid);

            results = JSON.parse(tmp{1}{1});
            
            names_corrected = fieldnames(results);
            names_corrected = names_corrected{1};
   
            results=getfield(results,names_corrected);
 
            for i=1:length(results)
                run = results{i};

                r = zeros(size(run.res,2), size(cell2mat(run.res{1}),2));
                for j = 1:length(run.res)
                    r(j,:) = cell2mat(run.res{j});
                end

                run.res = r;
                results{i} = run;
            end
            
            resultSaveName = strcat (saveLocation,'/', ...
                names{k},'_',tracker);
            save(resultSaveName, 'results');
        end
    end
end


end

