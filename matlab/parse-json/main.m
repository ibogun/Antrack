

saveRoot = '/Users/Ivan/Code/Libraries/matlab_libraries/tracker_benchmark_v1.0/results/';
loadRoot = '/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/results/';

conference = 'CVPR13';
trackerName = {'MBD32L0_1'};
%trackerName = {'RobStruck'};
%evalType = {'OPE','SRE', 'TRE'};
evalType={'OPE'};
names = getSequenceNamesOTB50();
% for loop -> tracker
for trackerID = 1: length(trackerName) % len(trackerName)
    
    display(strcat('Working with the tracker: ', trackerName{trackerID}));
    
    % for loop saveLocations & evalType
    for evalTypeID = 1 : length(evalType) % len(evalType)
        
        display(strcat('Experiment type: ', evalType{evalTypeID}));

        tracker = trackerName{trackerID};
        
        if strcmp(evalType{evalTypeID},'OPE')
            saveLocation =strcat(saveRoot, 'results_', evalType{evalTypeID});
        else
             saveLocation = strcat(saveRoot, 'results_', evalType{evalTypeID},...
             '_', conference);
        end

          % for loop name
        for k = 1: length(names)
            
            display(strcat('Sequence name: ', names{k}));

            filename = strcat(loadRoot,'/' ,evalType{evalTypeID},'/', tracker,...
                   '/', names{k},'.json');
               
            fid = fopen(filename,'rt');
            tmp = textscan(fid,'%s','Delimiter','\n', 'BufSize', 12312390);
            fclose(fid);

            results = JSON.parse(tmp{1}{1});
            
            names_corrected = names{k};
            names_corrected =strrep(names_corrected,'-','_');
            z = fieldnames(results);
            if(isfield(results,z))
                                    results=getfield(results,z{1});
            end
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

