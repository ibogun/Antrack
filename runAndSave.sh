#!/usr/bin/env bash


currentDir=$(pwd)


# cd build/
# make

#cd $currentDir

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform
	savePath=/Users/Ivan/Files/Results/Tracking/wu2013/
else
	# consider it linux
	savePath=/udrive/student/ibogun2010/Research/Results/wu2013/
fi

echo "Let's copy *.dat files from remote..."
scp ibogun2010@blueshark.fit.edu:/udrive/student/ibogun2010/Research/Results/wu2013/*.dat $savePath

scp ibogun2010@blueshark.fit.edu:/udrive/student/ibogun2010/Research/Results/wu2013/tracker_info.txt $savePath

datasetType='wu2013'


echo "Please enter run identifier..."
read runIdentifier

#./build/bin/robust_struck_tracker_v1.0

cd python/Evaluation/

echo $savePath
echo $datasetType
echo $runIdentifier

python generatePythonFilePickle.py $savePath $datasetType $runIdentifier

echo "Pickle file was generated"
cd $currentDir


echo "Lets regenerate plots for tracker evaluation..."

cd python/Evaluation
python DatasetEvaluation.py

cd $currentDir


# Now generate plots based on the pickle files, saves them to images and copies to the website page.
