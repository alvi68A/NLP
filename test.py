import torch
import transformers
from transformers import BertForSequenceClassification
import numpy as np

#Setting up device, model and tokenizer information
device = torch.device('cpu')
bert = "TurkuNLP/bert-base-finnish-cased-v1"
model_state_dict = torch.load("pytorch_model.bin", map_location=torch.device('cpu'))
model = BertForSequenceClassification.from_pretrained(bert, state_dict=model_state_dict, num_labels=5, ignore_mismatched_sizes=True)
tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

#Getting user inputted message
input_text = str(input("Enter phrase: "))

#Tokenizing and processing text
text = tokenizer(input_text, return_tensors="pt")
ids=text['input_ids']
attention_mask=text['attention_mask']

#Putting data to model and getting the predictions
logits=(model(ids, attention_mask=attention_mask).logits)

#Doing some tensor to numpy to list magic
final = logits.detach().numpy()
wastebin = final.copy()
final=final.tolist()
final=final[0]

#Getting largest number
largest_num = max(final)

#Printing results out
if final[0]==largest_num:
    print("Your input is in the most negative category")
    print("Your input was:", input_text)

elif final[1]==largest_num:
    print("Your input is in the negative category")
    print("Your input was:", input_text)

elif final[2]==largest_num:
    print("Your input is in the neutral category")
    print("Your input was:", input_text)

elif final[3]==largest_num:
    print("Your input is in the positive category")
    print("Your input was:", input_text)

elif final[4]==largest_num:
    print("Your input is in the most positive category")
    print("Your input was:", input_text)
