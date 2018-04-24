
## Code Notation 

### Data proprocessing

The data preprocessing was accomplished in scala 2.2.0. The codes could be found in the main file of data_preprocess/code/src while the library dependency would be found in the build.sbt. 

(i) Download DIAGNOSES_ICD.csv and NOTEEVENTS.csv from MIMICIII database (requested from https://mimic.physionet.org/) and locate it into data_preprocess/Code/data.

(ii) Run data_preprocssing/Code/src/main/scala/BDH/main.scala in IntelliJ. 

(iii) Locate the summarized output for 18 categorical ICD9 codes in data_preprocessing/Code/data/NOTES_ALL_ICD9.csv, for top 20 generic ICD9 codes in data_preprocessing/Code/data/NOTES_TOP20_ICD9.csv

(iv) Visualize the summarized data in R. 

### Model Training and Testing

Platform: Python 3.6.3

#### Model Preparation

(i) Download pre-trained GloVe model Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download) from https://nlp.stanford.edu/projects/glove/;

(ii) Unzip the glove.840B.300d.zip into the /training_model/inputData;

(iii) Download NOTES_18Categories_ICD9.csv and NOTES_TOP20_ICD9.csv form https://drive.google.com/drive/u/1/folders/1PSjmBQcfvhVWgEAjAP8367UJ7cDds3VA to /training_model/inputData;

#### Parameter Setting

(i) Open /training_model/GloVe/multiclass_multilabel.py; 

(ii) Set the input parameters, including the "category_number", "dataset_path", "model_name", "max_sequence_len", "pre_train" and "output_path" for the model.For top-level 18 categories codes, "category_number" equals 18 and "dataset_path" equals "../inputData/NOTES_18Categories_ICD9.csv"; For top 20 generic codes,"category_number" equals 20 and "dataset_path" equals "../inputData/NOTES_TOP20_ICD9.csv". Notice that the model_name could be either "LSTM_MODEL" or "CNN_MODEL", "max_sequence_len" could be any preferred word sequence length. "pre-train" means loading the trained model weights, "output_path" is to store the output of the trained model and the testing. 

#### Model training and testing

(i) Run /training_model/GloVe/multiclass_multilabel.py;

(ii) Find the model training curves (e.g. accuracy.png and loss.png) and weights in the folder of output_path;

(iii) Find the Recall, precision, F1 score and accuracy for model testing in the standard output.
