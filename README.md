# Project README file

## Project Description 


## Code Notation 

### Data proprocessing

The data preprocessing was accomplished in scala. The codes could be found in the main file of data_preprocess/code/src while the library dependency would be found in the build.sbt. The original data file DIAGNOSES_ICD and NOTEEVENTS were download from MIMICiIII database and located into the data folder underneath the data_preprocess/code folder.

The ICD9 codes from DIAGNOSES_ICD were categorized by the first three digits to obtain the 18 categorical ICD9 codes and top20 generic ICD9 codes. Meanwhile, statistical description including the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were summarized. Then the table from NOTEEVENTS was joined with the categorical ICD9 codes and top20 generic ICD9 codes by HAMD_ID (or admission ID). The obtained tables of clinical notes with categorical ICD9 codes and top20 generic ICD9 codes were exported in CSV as input for further NLP and deep learning procedures. The statistical summary of the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were also exported in CSV for data visualization in R. 
