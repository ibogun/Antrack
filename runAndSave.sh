

currentDir=$(pwd)


cd build/
make

cd $currentDir

if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform    
	savePath="/Users/Ivan/Files/Results/Tracking/wu2013"  
else
	# consider it linux
	savePath="/media/drive/UbuntuFiles/Results/wu2013"
fi


datasetType='wu2013'


echo "Please enter run identifier..."
read runIdentifier

./build/bin/robust_struck_tracker_v1.0

cd python/Evaluation/

echo $savePath
echo $datasetType
echo $runIdentifier

python generatePythonFilePickle.py $savePath $datasetType $runIdentifier

cd $currentDir


