## Project Description 


## Code Notation 

### Data proprocessing

The data preprocessing was accomplished in scala. The codes could be found in the main file of data_preprocess/code/src while the library dependency would be found in the build.sbt. The original data file DIAGNOSES_ICD and NOTEEVENTS were download from MIMICIII database and located into the data folder underneath the data_preprocess/code folder.

The ICD9 codes from DIAGNOSES_ICD were categorized by the first three digits to obtain the 18 categorical ICD9 codes and top20 generic ICD9 codes. Meanwhile, statistical description including the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were summarized. Then the table from NOTEEVENTS was joined with the categorical ICD9 codes and top20 generic ICD9 codes by HAMD_ID (or admission ID). The obtained tables of clinical notes with categorical ICD9 codes and top20 generic ICD9 codes were exported in CSV as input for further NLP and deep learning procedures. The statistical summary of the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were also exported in CSV for data visualization in R. 

After running the code in main file, all the generated CSV files could be found in the data_preprocess/code/data folder. 

### Model Training and Testing
#### (1)Model Preparation
(i)Download pre-trained GloVe model Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from https://nlp.stanford.edu/projects/glove/; 
(ii)Unzip the downloaded glove.840B.300d.zip into the /training_model/inputData; 
(iii)Copy the corresponding .csv file from /data_preprocessing/data and paste it into the /training_model/inputData;
#### (2) Parameter Setting
(i) Open /training_model/GloVe/multiclass_multilabel.py; 
(ii) Set the input parameters, including the "model_name", max_sequence_len, pre_train, output_path, for the model; Notice that the model_name could be either "LSTM_MODEL" or "CNN_MODEL", "max_sequence_len" is the preferred word sequence length, "pre-train" means loading the trained model weights, "output_path" is to store the output from the model training and testing; 
#### (3) Model training and testing
(i) Run /training_model/GloVe/multiclass_multilabel.py; 
(ii) Find the model training curves and weights in the folder of output_path; 
(iii) Find the Recall, precision, F1 score and accuracy for model testing in python console output.
