import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
from main import evaluate, train_dataloader, val_dataloader, test_dataloader

bert = "TurkuNLP/bert-base-finnish-cased-v1"
model_state_dict = torch.load("pytorch_model.bin")
model = BertForSequenceClassification.from_pretrained(bert, state_dict=model_state_dict, num_labels=6, return_dict=False)
device=torch.device('cuda:0')

model.to(device)
model.eval()

_, train_correct, train_predicted = evaluate(model, train_dataloader)
_, dev_correct, dev_predicted = evaluate(model, val_dataloader)
_, test_correct, test_predicted = evaluate(model, test_dataloader)

print("Training performance", precision_recall_fscore_support(train_correct, train_predicted, average="micro"))
print("Development performance", precision_recall_fscore_support(dev_correct, dev_predicted, average="micro"))
print("Test performance", precision_recall_fscore_support(test_correct, test_predicted, average="micro"))

bert_accuracy = np.mean(test_predicted == test_correct)

print(classification_report(test_correct, test_predicted))
