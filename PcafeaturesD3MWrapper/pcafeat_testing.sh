#!/bin/bash -e 

cd /primitives
git checkout simon_pipelines
#git remote add upstream https://gitlab.com/datadrivendiscovery/primitives
#git pull upstream master

Datasets=('1491_one_hundred_plants_margin' 'LL0_1100_popularkids')
rm /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines/test_pipeline/*
cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines
#mkdir test_pipeline
#mkdir experiments
#cd test_pipeline
# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt
#cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.rffeatures.Rffeatures/3.1.1/pipelines
#mkdir test_pipeline
#mkdir experiments
#cd test_pipeline
# create text file to record scores and timing information
#touch scores.txt
#echo "DATASET, F1 SCORE, EXECUTION TIME" >> scores.txt

match="step_1.add_output('produce')"
insert="Temporary Line threshold"
file="/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py"
sed -i "s/$match/$match\n$insert/" $file
#insert="Temporary Line num_features"
#file="/src/rffeaturesd3mwrapper/RffeaturesD3MWrapper/python_pipeline_generator_rffeatures.py"
#sed -i "s/$match/$match\n$insert/" $file

for i in "${Datasets[@]}"; do
  cd /primitives/v2019.6.7/Distil/d3m.primitives.feature_selection.pca_features.Pcafeatures/3.0.2/pipelines/test_pipeline
  best_score=0
  for n in $(seq 0 0.1 0.95); do
    file="/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py"
    sed -i '/threshold/d' $file
    insert="step_1.add_hyperparameter(name='threshold', argument_type=ArgumentType.VALUE,data=$n)"
    sed -i "s/$match/$match\n$insert/" $file
    # generate and save pipeline + metafile
    python3 "/src/pcafeaturesd3mwrapper/PcafeaturesD3MWrapper/python_pipeline_generator_pcafeatures.py" $i
    
    # test and score pipeline
    start=`date +%s` 
    python3 -m d3m runtime -d /datasets/ fit-score -m *.meta -p *.json -c scores.csv
    end=`date +%s`
    runtime=$((end-start))

    # copy pipeline if execution time is less than one hour
    if [ $runtime -lt 3600 ]; then
      echo "$i took less than 1 hour, evaluating pipeline"
      IFS=, read col1 score col3 col4 < <(tail -n1 scores.csv)
      echo "$score"
      echo "$best_score"
      if [[ $score > $best_score ]]; then
        echo "$i, $score, $runtime" >> scores.txt
        best_score=$score
        echo "$best_score"
        rm ../experiments/*
        cp *.meta ../experiments/
        cp *.json ../experiments/
       fi
    fi
  
  # cleanup temporary file
  rm *.meta
  rm *.json
  rm scores.csv
  done 
done

