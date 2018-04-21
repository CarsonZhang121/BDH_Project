# Project README file

## Project Description 


### Data proprocessing

The data preprocessing was accomplished in scala. The codes could be found in the main file of code folder while the library dependency would be found in the build.sbt. 
The original data file DIAGNOSES_ICD could be imported into the data folder underneath the code foloder and the ICD9 codes were categorized by the first three digits to obtain the 18 categorical ICD9 codes and top20 generic ICD9 codes. Meanwhile, statistical description including the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were summarized. Then the data file NOTEEVENTS was imported and joined with the categorical ICD9 codes and top20 generic ICD9 codes by HAMD_ID (or admission ID). The obtained tables of clinical notes with categorical ICD9 codes and top20 generic ICD9 codes were exported in CSV as input for further NLP and deep learning procedures. The statistical summary of the frequency of 18 categorical ICD9 codes and top20 generic ICD9 codes per admission were also exported in CSV for data visualization in R. 
