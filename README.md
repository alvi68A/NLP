# NLP
Natural Language Processing Model for finnish Twitter discourse. Uses TurkuNLP's pretrained FinnBert model with FinnSentiment dataset. 

FinnSentiment dataset provided by Language Bank of Finland derived positive negative and neutral scores to 1-5 scale which was used for this model. 
In the data, 1 is the most negative sentiment and 5 is the most positive sentiment. 


# Training scores:
Training performance: (0.9215059957061319)
Development performance: (0.9283619606200252)
Test performance: (0.9153749476330122)


              precision    recall  f1-score   support
           1       0.87      0.75      0.81       114
           2       0.89      0.90      0.90       568
           3       0.95      0.96      0.95      1266
           4       0.86      0.85      0.85       307
           5       0.88      0.86      0.87       132
           
           
     accuracy                           0.92      2387
     m avg          0.89      0.87      0.88      2387
     w avg          0.91      0.92      0.91      2387


# Trying the model
If you want to try the model:

  * first run main.py
  * let the model train
  * run test.py with your own data


# Additional model: languageDetector
english and finish language detector, which achieved 99,5% accuracy over the testing dataset
