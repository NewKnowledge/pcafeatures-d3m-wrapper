#!/bin/bash -e 

cd /primitives
git checkout simon_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

Datasets=('1491_one_hundred_plants_margin' 'LL0_1100_popularkids')
rm /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines/test_pipeline/*
cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines
#mkdir test_pipeline
cd test_pipeline

# create text file to record scores and timing information
touch scores.txt
echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

for i in "${Datasets[@]}"; do

  # generate and save pipeline + metafile
  python3 "/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py" $i

  # test and score pipeline
  start=`date +%s` 
  python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
  end=`date +%s`
  runtime=$((end-start))

  # copy pipeline if execution time is less than one hour
  if [ $runtime -lt 3600 ]; then
     echo "$i took less than 1 hour, copying pipeline"
     cp * ../
  fi

  # save information
  IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
  echo "$i, $score, $runtime" >> scores.txt
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
done
