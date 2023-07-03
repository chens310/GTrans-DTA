# GTrans-DTA: A transformer network incorporating graph spatial structure information for drug–target binding affinity prediction


## dependencies

pytorch == 1.11.0
python == 3.8.10
rdkit == 2022.09.5
scipy ==1.10.1
networkx == 2.8.8 
pandas == 2.0.0 
fair-esm == 2.0.0

## Data Preprocess
You first need to generate the  5-fold cross-validation data  from raw data.   
Run command as follows:
    
    python datahelper.py
    

Then you will get the following files:
1. data/davis/davis_train_fold0-5.csv
2. data/davis/davis_test_fold0-5.csv
3. data/kiba/kiba_train_fold0-5.csv
4. data/kiba/kiba_test_fold0-5.csv
 
you will get protein represent and protein PDB files from pretraining model namely esm-1b and esm-fold:
Run command as follows:
    
    python protein_feature.py

## Training
First you should create a new folder for saved models, path = "GTrans-DTA/saved_models".  
Then run command below to train FusionDTA.

    python training.py

## testing
Run command below for validation.

    python validating.py 


