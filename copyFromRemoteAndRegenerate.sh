currentDir=$(pwd)

echo "Please enter the name or wildcard to use to copy files from remote (ms_hog or ms*)"

read copyWildcard

echo "Copying pickles from remote..."


scp ibogun2010@blueshark.fit.edu:/udrive/student/ibogun2010/Research/Code/Antrack/python/Evaluation/Runs/${copyWildcard}.p ./python/Evaluation/Runs/



echo "Lets regenerate plots for tracker evaluation..."

cd python/Evaluation
python DatasetEvaluation.py $copyWildcard

cd $currentDir


# Now generate plots based on the pickle files, saves them to images and copies to the website page.
