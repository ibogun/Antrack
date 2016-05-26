function [names] = getSequenceNamesFast(  )
%GETSEQUENCENAMES Summary of this function goes here
%   Detailed explanation goes here

seqs=configSeqs;

names = cell(length(seqs),1);

for i =1:length(seqs)
    names{i} = seqs{i}.name;
end

end


function seqs=configSeqs

seqVTD={     struct('name','Ironman','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/ironman/','startFrame',1,'endFrame',166,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','Deer','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/deer/','startFrame',1,'endFrame',71,'nz',4,'ext','jpg','init_rect', [0,0,0,0])};

seqOther={     struct('name','MotorRolling','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/motorRolling/','startFrame',1,'endFrame',164,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
    struct('name','MountainBike','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/mountainBike/','startFrame',1,'endFrame',228,'nz',4,'ext','jpg','init_rect', [0,0,0,0])};
   

seqNew={struct('name','Biker','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/biker/','startFrame',1,'endFrame',142','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
            struct('name','Bird2','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/bird2/','startFrame',1,'endFrame',99','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','Dancer','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/dancer/','startFrame',1,'endFrame',225','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
    struct('name','Dancer2','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/dancer2/','startFrame',1,'endFrame',150','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','DragonBaby','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/dragonbaby/','startFrame',1,'endFrame',113','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','Skiing','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/skiing/','startFrame',1,'endFrame',81,'nz',4,'ext','jpg','init_rect', [0,0,0,0]),...
        struct('name','Human8','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/human8/','startFrame',1,'endFrame',128','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','Jump','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/jump/','startFrame',1,'endFrame',122','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
    struct('name','KiteSurf','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/kitesurf/','startFrame',1,'endFrame',84','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
    struct('name','Man','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/man/','startFrame',1,'endFrame',134','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','Skater','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/skater/','startFrame',1,'endFrame',160','nz',4,'ext','jpg','init_rect',[0,0,0,0]),...
        struct('name','Trans','path','/Users/Ivan/Code/Tracking/Antrack/python/visual-tracking-benchmark/data/trans/','startFrame',1,'endFrame',124','nz',4,'ext','jpg','init_rect',[0,0,0,0])};
    

seqs=[seqVTD,seqOther,seqNew];

end