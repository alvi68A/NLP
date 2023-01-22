import torch
import time
import datetime
import random
import numpy as np
import csv
import pandas as pd
import tqdm
from tqdm import trange
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
!pip install transformers
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup


#Setting the device and activating GPU
device = torch.device('cuda:0')
df = pd.read_csv('cleaned.csv')


#Dividing dataset
train = df.sample(frac =.80)
val = df.sample(frac =.10)
test = df.sample(frac=.10)


#Dividing labels and features
x_train = train.iloc[:,1]
y_train = train.iloc[:,0]

x_val=val.iloc[:,1]
y_val=val.iloc[:,0]

x_test=pd.DataFrame(test.iloc[:,1])
y_test=test.iloc[:,0]


#Listing values for tokenization
x_test=x_test.values.tolist()
x_train=x_train.values.tolist()
x_val=x_val.values.tolist()


#Creating empty variables for tokenized values
x_val_tokenized = []
x_train_tokenized = []
x_test_tokenized = []


#Tokenizer finetuning 
tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")


#Tokenizer iterations
for i in x_train:
    x_train_tokenized.append(tokenizer(i, truncation=True, padding='max_length', max_length=512))

for i in x_val:
    x_val_tokenized.append(tokenizer(i, truncation=True, padding='max_length', max_length=512))

for i in x_test:
    x_test_tokenized.append(tokenizer(i, truncation=True, padding='max_length', max_length=512))

    
#Creating empty lists for mask and id division
train_seq = []
train_mask =[]
val_seq = []
val_mask = []
test_seq = []
test_mask = []


#Dividing masks and ids
for i in range(len(x_train_tokenized)):
    train_seq.append(x_train_tokenized[i]['input_ids'])
    train_mask.append(x_train_tokenized[i]['attention_mask'])

for i in range(len(x_val_tokenized)):
    val_seq.append(x_val_tokenized[i]['input_ids'])
    val_mask.append(x_val_tokenized[i]['attention_mask'])

for i in range(len(x_test_tokenized)):
    test_seq.append(x_test_tokenized[i]['input_ids'])
    test_mask.append(x_test_tokenized[i]['attention_mask'])

    
#Tensoring values
train_seq = torch.tensor(train_seq)
train_mask = torch.tensor(train_mask)
train_y = torch.tensor(y_train.values.tolist())

val_seq = torch.tensor(val_seq)
val_mask = torch.tensor(val_mask)
val_y = torch.tensor(y_val.values.tolist())

test_seq = torch.tensor(test_seq)
test_mask = torch.tensor(test_mask)
test_y = torch.tensor(y_test.values.tolist())

#Defining model
model = transformers.BertForSequenceClassification.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", num_labels = 6, return_dict=False)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

   
#Defining batch size
batch_size = 16


#Making dataloader and defining TensorDataset
train_data = TensorDataset(train_seq, train_mask, train_y)
train_dataloader=DataLoader(
            train_data,
            sampler=RandomSampler(train_data),
            batch_size=batch_size,
            num_workers=0
        )
val_data = TensorDataset(val_seq, val_mask, val_y)
val_dataloader= DataLoader(
            val_data,
            batch_size=batch_size,
            num_workers=0
        )

test_data = TensorDataset(test_seq, test_mask, test_y)


#Defining epochs and calculating total steps
epochs = 6
total_steps = len(train_dataloader) * epochs
steps_per_epoch = len(x_train)  // batch_size
total_training_steps = steps_per_epoch * epochs
warmup_steps = total_training_steps // 5


#Seeding all seed values for future applications
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#Defining optimizer and scheduler
optimizer = AdamW(model.parameters(),
                  lr= 2e-5,
                  correct_bias=False
                  )
scheduler= get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )


#Defining evaluation function
def evaluate(model, dataloader):
    print("\nEvaluating...")
    
    #Putting model to evaluation mode
    model.eval()
    
    #zeroing values
    nb_eval_steps = 0
    eval_loss = 0
    predicted_labels, correct_labels = [],[]
    
    #iterating dataloader
    for step,batch in enumerate(dataloader):
        if step % 100 == 0 and not step == 0:
            print(' Batch {:>5,} of {:>5,}.'.format(step, len(dataloader)))
        batch = tuple(t.to(device) for t in batch)
        sent_id, mask, labels = batch

        with torch.no_grad():
            loss, logits = model(sent_id, attention_mask = mask, labels=labels)
        
        outputs = np.argmax(logits.to('cpu'), axis=1)
        labels= labels.to('cpu').numpy()
        predicted_labels += list(outputs)
        correct_labels +=list(labels)
        nb_eval_steps += 1
        eval_loss += loss.mean().item()
        

    #Calculating loss and creating labels lists
    eval_loss = eval_loss / nb_eval_steps
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)

    #returning loss, correct_labels and predictions
    return eval_loss, correct_labels, predicted_labels
  
  
#Setting up training parameters
MODEL_FILE_NAME = "pytorch_model.bin"
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 5
PATIENCE = 3
loss_history = []
no_improvement = 0


#Training iterations
for _ in trange(int(epochs), desc="Epoch"):
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step,batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
        if step % 100 == 0 and not step == 0:
            print(' Batch {:>5,} of {:>5,}.'.format(step, len(train_dataloader)))

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch
        outputs = model(sent_id, attention_mask=mask, labels=labels)
        loss=outputs[0]

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS


        total_loss += loss.item()
        loss.backward()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    dev_loss, _, _ = evaluate(model, val_dataloader)
    print("Loss history:", loss_history)
    print("Val loss:", dev_loss)

    if len(loss_history) == 0 or dev_loss < min(loss_history):
        no_improvement = 0
        model_to_save= model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), MODEL_FILE_NAME)
    else:
        no_improvement += 1

    if no_improvement >= PATIENCE:
        print("No improvement on developement set. Finish training.")
        break

    loss_history.append(dev_loss)
