## Project Description 


## Code Notation 

### Data proprocessing

The data preprocessing was accomplished in scala. The codes could be found in the main file of data_preprocess/code/src while the library dependency would be found in the build.sbt. The original data file DIAGNOSES_ICD and NOTEEVENTS were download from MIMICIII database and located into the data folder underneath the data_preprocess/code folder.

The ICD9 codes from DIAGNOSES_ICD were categorized by the first three digits to obtain the 18 categorical ICD9 codes and top20 generic ICD9 codes. Meanwhile, statistical description including the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were summarized. Then the table from NOTEEVENTS was joined with the categorical ICD9 codes and top20 generic ICD9 codes by HAMD_ID (or admission ID). The obtained tables of clinical notes with categorical ICD9 codes and top20 generic ICD9 codes were exported in CSV as input for further NLP and deep learning procedures. The statistical summary of the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were also exported in CSV for data visualization in R. After running the code in main file, all the generated CSV files could be found in the data_preprocess/code/data folder. 

### Model Training and Testing
#### Model Preparation
Download pre-trained GloVe model Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from https://nlp.stanford.edu/projects/glove/; unzip the downloaded glove.840B.300d.zip into the "inputData" folder;
#### Parameter Setting
Go to /training_model/GloVe and open multiclass_multilabel.py; Set the input parameters, including the "model_name", max_sequence_len, pre_train, output_path, for the model; Notice that the model_name could be either "LSTM_MODEL" or "CNN_MODEL", "max_sequence_len" is the preferred word sequence length, "pre-train" 
#### 
