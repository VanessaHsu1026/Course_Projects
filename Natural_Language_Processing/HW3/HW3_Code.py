# !pip install datasets==2.21.0
# !pip install torchmetrics
import transformers as T
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef, Accuracy, F1Score

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 有些中文的標點符號在tokenizer編碼以後會變成[UNK]，所以將其換成英文標點
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]

tokenizer = T.BertTokenizer.from_pretrained("blackbird/bert-base-uncased-MNLI-v1", cache_dir="./cache/") # Rank 1
# tokenizer = T.BertTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI", cache_dir="./cache/") # Rank 2
# tokenizer = T.BertTokenizer.from_pretrained("ishan/bert-base-uncased-mnli", cache_dir="./cache/") # Rank 3
# tokenizer = T.BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2", cache_dir="./cache/") # Rank 4
# tokenizer = T.BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/") # Rank 5
# tokenizer = T.BertTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-sst2", cache_dir="./cache/") # Rank 6
# tokenizer = T.BertTokenizer.from_pretrained("Intel/bert-base-uncased-mrpc", cache_dir="./cache/") # Rank 7
# tokenizer = T.BertTokenizer.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc", cache_dir="./cache/") # Rank 8
# tokenizer = T.BertTokenizer.from_pretrained("textattack/bert-base-uncased-MRPC", cache_dir="./cache/") # Rank 9
# tokenizer = T.BertTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news", cache_dir="./cache/") # Rank 10

class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, cache_dir="./cache/",trust_remote_code=True).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # 把中文標點替換掉
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# Define the hyperparameters
lr = 4e-5
epochs = 5
train_batch_size = 64
validation_batch_size = 64
test_batch_size = 64

# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.

# TODO1-1: Implement the collate_fn function
def collate_fn(batch):
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.

    sentences = [item["premise"] + " " + item["hypothesis"] for item in batch]
    relatedness_scores = [item["relatedness_score"] for item in batch]
    entailment_judgments = [item["entailment_judgment"] for item in batch]

    # Tokenize and pad sequences
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Convert labels to tensor
    relatedness_scores = torch.tensor(relatedness_scores, dtype=torch.float)
    entailment_judgments = torch.tensor(entailment_judgments, dtype=torch.float)

    return tokenized_inputs, relatedness_scores, entailment_judgments

# # Only for relatedness score task
# def collate_fn_relatedness(batch):
#     sentences = [item["premise"] + " " + item["hypothesis"] for item in batch]
#     relatedness_scores = [item["relatedness_score"] for item in batch]

#     tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#     relatedness_scores = torch.tensor(relatedness_scores, dtype=torch.float)
#     return tokenized_inputs, relatedness_scores

# # Only for entailment judgment task
# def collate_fn_entailment(batch):
#     sentences = [item["premise"] + " " + item["hypothesis"] for item in batch]
#     entailment_judgments = [item["entailment_judgment"] for item in batch]

#     tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
#     entailment_judgments = torch.tensor(entailment_judgments, dtype=torch.long)
#     return tokenized_inputs, entailment_judgments

# Instantiate Datasets
train_dataset = SemevalDataset(split="train")
validation_dataset = SemevalDataset(split="validation")
test_dataset = SemevalDataset(split="test")

# TODO1-2: Define your DataLoader
dl_train = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
dl_validation = DataLoader(validation_dataset, batch_size=validation_batch_size, collate_fn=collate_fn, shuffle=False)
dl_test = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False)

# # Create DataLoader for each sub-task
# dl_train_relatedness = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn_relatedness, shuffle=True)
# dl_validation_relatedness = DataLoader(validation_dataset, batch_size=validation_batch_size, collate_fn=collate_fn_relatedness, shuffle=False)
# dl_test_relatedness = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn_relatedness, shuffle=False)
# dl_train_entailment = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn_entailment, shuffle=True)
# dl_validation_entailment = DataLoader(validation_dataset, batch_size=validation_batch_size, collate_fn=collate_fn_entailment, shuffle=False)
# dl_test_entailment = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn_entailment, shuffle=False)

# TODO2: Construct your model
import torch.nn as nn
from transformers import BertModel

class MultiLabelModel(nn.Module):
    def __init__(self):
        super(MultiLabelModel, self).__init__()
        # Load BERT model as the encoder
        self.bert = BertModel.from_pretrained("blackbird/bert-base-uncased-MNLI-v1", cache_dir="./cache/") # Rank 1
        # self.bert = BertModel.from_pretrained("textattack/bert-base-uncased-MNLI", cache_dir="./cache/") # Rank 2
        # self.bert = BertModel.from_pretrained("ishan/bert-base-uncased-mnli", cache_dir="./cache/") # Rank 3
        # self.bert = BertModel.from_pretrained("textattack/bert-base-uncased-SST-2", cache_dir="./cache/") # Rank 4
        # self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/") # Rank 5
        # self.bert = BertModel.from_pretrained("gchhablani/bert-base-cased-finetuned-sst2", cache_dir="./cache/") # Rank 6
        # self.bert = BertModel.from_pretrained("Intel/bert-base-uncased-mrpc", cache_dir="./cache/") # Rank 7
        # self.bert = BertModel.from_pretrained("google-bert/bert-base-cased-finetuned-mrpc", cache_dir="./cache/") # Rank 8
        # self.bert = BertModel.from_pretrained("textattack/bert-base-uncased-MRPC", cache_dir="./cache/") # Rank 9
        # self.bert = BertModel.from_pretrained("textattack/bert-base-uncased-ag-news", cache_dir="./cache/") # Rank 10

        # Define a dropout layer for regularization
        self.dropout = nn.Dropout(0.3)

        # Define output layers
        self.relatedness_score = nn.Linear(self.bert.config.hidden_size, 1)  # Continuous output for regression
        self.entailment_judgment = nn.Linear(self.bert.config.hidden_size, 3)  # 3-class classification output
    
    # Forward pass
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Get the [CLS] token's representation
        cls_output = outputs.pooler_output

        # Apply dropout
        cls_output = self.dropout(cls_output)

        # Predict relatedness score and entailment judgment
        relatedness = self.relatedness_score(cls_output)  # Regression output
        entailment = self.entailment_judgment(cls_output)  # Classification output (logits for 3 classes)

        return relatedness.squeeze(-1), entailment  # Squeeze only the relatedness output

# # Define separate models for each sub-task
# class RelatednessModel(nn.Module):
#     def __init__(self):
#         super(RelatednessModel, self).__init__()
#         self.bert = BertModel.from_pretrained("blackbird/bert-base-uncased-MNLI-v1", cache_dir="./cache/")
#         self.dropout = nn.Dropout(0.3)
#         self.relatedness_score = nn.Linear(self.bert.config.hidden_size, 1)  # Regression output
    
#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         cls_output = self.dropout(outputs.pooler_output)
#         return self.relatedness_score(cls_output).squeeze(-1)

# class EntailmentModel(nn.Module):
#     def __init__(self):
#         super(EntailmentModel, self).__init__()
#         self.bert = BertModel.from_pretrained("blackbird/bert-base-uncased-MNLI-v1", cache_dir="./cache/")
#         self.dropout = nn.Dropout(0.3)
#         self.entailment_judgment = nn.Linear(self.bert.config.hidden_size, 3)  # Classification output
    
#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         cls_output = self.dropout(outputs.pooler_output)
#         return self.entailment_judgment(cls_output)

# Initialize Model
model = MultiLabelModel().to(device)

# TODO3: Define your optimizer and loss function

# import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         # Convert logits to probabilities using softmax
#         probs = F.softmax(logits, dim=1)
        
#         # One-hot encode the targets
#         targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 1)
        
#         # Compute the intersection and union
#         intersection = (probs * targets_one_hot).sum(dim=0)
#         union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
#         # Calculate Dice coefficient
#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
#         # Return the Dice loss
#         return 1 - dice.mean()

# TODO3-1: Define your Optimizer
optimizer = AdamW(model.parameters(), lr=lr)
# TODO3-2: Define your loss functions (you should have two)
regression_loss_fn = nn.MSELoss()       # For relatedness score (regression task)
classification_loss_fn = nn.CrossEntropyLoss()  # For entailment judgment (classification task)
# regression_loss_fn = nn.L1Loss()  # Replacing MSELoss with MAELoss
# classification_loss_fn = DiceLoss() # Replacing CrossEntropyLoss with DiceLoss

# scoring functions
spc = SpearmanCorrCoef()
acc = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3, average='macro')

# I used ChatGPT to assist me in writing code and debugging.

import os
# Create 'saved_models' directory if it doesn't exist
os.makedirs('./saved_models', exist_ok=True)

# TODO4: Write the training loop
for ep in range(epochs):
    model.train()
    total_train_loss = 0
    pbar_train = tqdm(dl_train, desc=f"Training epoch [{ep+1}/{epochs}]")
    for batch in pbar_train:
        # Unpack the batch
        tokenized_inputs, relatedness_scores, entailment_judgments = batch
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
        token_type_ids = tokenized_inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Clear gradient
        optimizer.zero_grad()
        # Forward pass
        relatedness_preds, entailment_preds = model(input_ids, attention_mask, token_type_ids)

        # Compute loss
        relatedness_loss = regression_loss_fn(relatedness_preds, relatedness_scores.to(device))
        entailment_loss = classification_loss_fn(entailment_preds, entailment_judgments.to(device).long())
        loss = relatedness_loss + entailment_loss
        total_train_loss += loss.item()

        # Back-propagation
        loss.backward()
        optimizer.step()
        
        # Update progress bar with loss
        pbar_train.set_postfix(loss=f"{loss.item():.2f}")

    # Calculate average training loss
    avg_train_loss = total_train_loss / len(dl_train)

    # TODO5: Write the evaluation loop
    model.eval()
    total_val_loss = 0
    total_spc = 0
    total_acc = 0
    total_f1 = 0
    prediction_samples = []

    pbar_val = tqdm(dl_validation, desc=f"Validation epoch [{ep+1}/{epochs}]")
    with torch.no_grad():
        for batch in pbar_val:
            # Unpack the batch
            tokenized_inputs, relatedness_scores, entailment_judgments = batch
            input_ids = tokenized_inputs['input_ids'].to(device)
            attention_mask = tokenized_inputs['attention_mask'].to(device)
            token_type_ids = tokenized_inputs.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Forward pass
            relatedness_preds, entailment_preds = model(input_ids, attention_mask, token_type_ids)

            # Classification Predictions
            entailment_preds_class = torch.argmax(entailment_preds, dim=-1)

            # Store misclassified samples
            for idx, (true_label, pred_label, rel_pred, rel_true) in enumerate(
                    zip(entailment_judgments, entailment_preds_class.cpu(), relatedness_preds, relatedness_scores)):
                dynamic_threshold = 0.1 * rel_true  # 10% of the true value
                if true_label != pred_label or abs(rel_pred - rel_true) > dynamic_threshold:
                    prediction_samples.append({
                        "input_sentences": tokenizer.decode(input_ids[idx], skip_special_tokens=True),
                        "true_entailment": true_label.item(),
                        "predict_entailment": pred_label.item(),
                        "true_relatedness": rel_true.item(),
                        "predict_relatedness": rel_pred.item()
                    })

            # Compute loss
            relatedness_loss = regression_loss_fn(relatedness_preds, relatedness_scores.to(device))
            entailment_loss = classification_loss_fn(entailment_preds, entailment_judgments.to(device).long())
            loss = relatedness_loss + entailment_loss
            total_val_loss += loss.item()

            # Compute evaluation metrics
            spc_score = spc(relatedness_preds.cpu(), relatedness_scores)
            acc_score = acc(entailment_preds.cpu(), entailment_judgments)
            f1_score = f1(entailment_preds.cpu(), entailment_judgments)

            # Accumulate metrics
            total_spc += spc_score.item()
            total_acc += acc_score.item()
            total_f1 += f1_score.item()

    # Calculate average validation metrics
    avg_val_loss = total_val_loss / len(dl_validation)
    avg_spc = total_spc / len(dl_validation)
    avg_acc = total_acc / len(dl_validation)
    avg_f1 = total_f1 / len(dl_validation)

    # Print results for the epoch
    print(f"\nEpoch [{ep+1}/{epochs}]")
    print(f"Training Loss: {avg_train_loss:.2f}")
    print(f"Validation Loss: {avg_val_loss:.2f}")
    print(f"SpearmanCorrCoef: {avg_spc:.2f}, Accuracy: {avg_acc:.2f}, F1 Score: {avg_f1:.2f}\n")
   
    # Save model checkpoint for each epoch
    torch.save(model.state_dict(), f'./saved_models/ep{ep+1}.ckpt')

import pandas as pd

pd.DataFrame(prediction_samples).to_csv("prediction_samples.csv", index=False)

"""For test set predictions, you can write perform evaluation similar to #TODO5."""

# Write the testing loop
model.eval()
total_test_loss = 0
total_spc = 0
total_acc = 0
total_f1 = 0

pbar_test = tqdm(dl_test, desc=f"Testing")
with torch.no_grad():
    for batch in pbar_test:
        # Unpack the batch
        tokenized_inputs, relatedness_scores, entailment_judgments = batch
        input_ids = tokenized_inputs['input_ids'].to(device)
        attention_mask = tokenized_inputs['attention_mask'].to(device)
        token_type_ids = tokenized_inputs.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Forward pass
        relatedness_preds, entailment_preds = model(input_ids, attention_mask, token_type_ids)

        # Compute loss
        relatedness_loss = regression_loss_fn(relatedness_preds, relatedness_scores.to(device))
        entailment_loss = classification_loss_fn(entailment_preds, entailment_judgments.to(device).long())
        loss = relatedness_loss + entailment_loss
        total_test_loss += loss.item()

        # Compute evaluation metrics
        spc_score = spc(relatedness_preds.cpu(), relatedness_scores)
        acc_score = acc(entailment_preds.cpu(), entailment_judgments)
        f1_score = f1(entailment_preds.cpu(), entailment_judgments)

        # Accumulate metrics
        total_spc += spc_score.item()
        total_acc += acc_score.item()
        total_f1 += f1_score.item()

# Calculate average testing metrics
avg_test_loss = total_test_loss / len(dl_test)
avg_spc = total_spc / len(dl_test)
avg_acc = total_acc / len(dl_test)
avg_f1 = total_f1 / len(dl_test)

# Print results for the epoch
print(f"Testing Loss: {avg_test_loss:.2f}")
print(f"SpearmanCorrCoef: {avg_spc:.2f}, Accuracy: {avg_acc:.2f}, F1 Score: {avg_f1:.2f}\n")

"""Train separate models for each sub-task"""
# # Relatedness Score Model 
# relatedness_model = RelatednessModel().to(device)
# optimizer_relatedness = AdamW(relatedness_model.parameters(), lr=lr)
# regression_loss_fn = nn.MSELoss()

# # Training loop for relatedness score
# for ep in range(epochs):
#     relatedness_model.train()
#     total_train_loss = 0
#     pbar_train = tqdm(dl_train_relatedness, desc=f"Training epoch [{ep+1}/{epochs}]")
#     for batch in pbar_train:
#         # Unpack the batch
#         tokenized_inputs, relatedness_scores = batch
#         input_ids = tokenized_inputs['input_ids'].to(device)
#         attention_mask = tokenized_inputs['attention_mask'].to(device)
#         # Clear gradient
#         optimizer_relatedness.zero_grad()
#         # Forward pass
#         preds = relatedness_model(input_ids, attention_mask)
#         # Compute loss
#         loss = regression_loss_fn(preds, relatedness_scores.to(device))
#         # Calculate total training loss
#         total_train_loss += loss.item()
#         # Back-propagation
#         loss.backward()
#         optimizer_relatedness.step()
#         # Update progress bar with loss
#         pbar_train.set_postfix(loss=f"{loss.item():.2f}")
#     # Calculate average training loss
#     avg_train_loss = total_train_loss / len(dl_train_relatedness)

#     # Validation loop for relatedness score
#     relatedness_model.eval()
#     total_val_loss = 0
#     total_spc = 0
#     pbar_val = tqdm(dl_validation_relatedness, desc=f"Validation epoch [{ep+1}/{epochs}]")
#     with torch.no_grad():
#         for batch in pbar_val:
#             # Unpack the batch
#             tokenized_inputs, relatedness_scores = batch
#             input_ids = tokenized_inputs['input_ids'].to(device)
#             attention_mask = tokenized_inputs['attention_mask'].to(device)

#             # Forward pass
#             relatedness_preds = relatedness_model(input_ids, attention_mask)
#             # Compute loss
#             relatedness_loss = regression_loss_fn(relatedness_preds, relatedness_scores.to(device))
#             total_val_loss += relatedness_loss.item()
#             # Compute evaluation metrics
#             spc_score = spc(relatedness_preds.cpu(), relatedness_scores)

#             # Accumulate metrics
#             total_spc += spc_score.item()

#     # Calculate average validation metrics
#     avg_val_loss = total_val_loss / len(dl_validation_relatedness)
#     avg_spc = total_spc / len(dl_validation_relatedness)

#     # Print results for the epoch
#     print(f"\nEpoch [{ep+1}/{epochs}]")
#     print(f"Relatedness Score Training Loss: {avg_train_loss:.2f}")
#     print(f"Relatedness Score Validation Loss: {avg_val_loss:.2f}")
#     print(f"SpearmanCorrCoef: {avg_spc:.2f}\n")

# # Testing loop for relatedness score
# relatedness_model.eval()
# total_test_loss = 0
# total_spc = 0
# pbar_test = tqdm(dl_test_relatedness, desc=f"Testing")
# with torch.no_grad():
#     for batch in pbar_test:
#         # Unpack the batch
#         tokenized_inputs, relatedness_scores = batch
#         input_ids = tokenized_inputs['input_ids'].to(device)
#         attention_mask = tokenized_inputs['attention_mask'].to(device)

#         # Forward pass
#         relatedness_preds = relatedness_model(input_ids, attention_mask)
#         # Compute loss
#         relatedness_loss = regression_loss_fn(relatedness_preds, relatedness_scores.to(device))
#         total_test_loss += relatedness_loss.item()
#         # Compute evaluation metrics
#         spc_score = spc(relatedness_preds.cpu(), relatedness_scores)

#         # Accumulate metrics
#         total_spc += spc_score.item()

# # Calculate average validation metrics
# avg_test_loss = total_test_loss / len(dl_test_relatedness)
# avg_spc = total_spc / len(dl_test_relatedness)

# # Print results for the epoch
# print(f"Relatedness Score Testing Loss: {avg_test_loss:.2f}")
# print(f"SpearmanCorrCoef: {avg_spc:.2f}\n")

# # Entailment Judgment Model 
# entailment_model = EntailmentModel().to(device)
# optimizer_entailment = AdamW(entailment_model.parameters(), lr=lr)
# classification_loss_fn = nn.CrossEntropyLoss()

# # Training loop for entailment judgment
# for ep in range(epochs):
#     entailment_model.train()
#     total_train_loss = 0
#     pbar_train = tqdm(dl_train_entailment, desc=f"Training epoch [{ep+1}/{epochs}]")
#     for batch in pbar_train:
#         # Unpack the batch
#         tokenized_inputs, entailment_judgments = batch
#         input_ids = tokenized_inputs['input_ids'].to(device)
#         attention_mask = tokenized_inputs['attention_mask'].to(device)
#         # Clear gradient
#         optimizer_entailment.zero_grad()
#         # Forward pass
#         preds = entailment_model(input_ids, attention_mask)
#         # Compute loss
#         loss = classification_loss_fn(preds, entailment_judgments.to(device))
#         # Calculate total training loss
#         total_train_loss += loss.item()
#         # Back-propagation
#         loss.backward()
#         optimizer_entailment.step()
#         # Update progress bar with loss
#         pbar_train.set_postfix(loss=f"{loss.item():.2f}")
#     # Calculate average training loss
#     avg_train_loss = total_train_loss / len(dl_train_entailment)

#     # Validation loop for entailment judgment
#     entailment_model.eval()
#     total_val_loss = 0
#     total_acc = 0
#     total_f1 = 0
#     pbar_val = tqdm(dl_validation_entailment, desc=f"Validation epoch [{ep+1}/{epochs}]")
#     with torch.no_grad():
#         for batch in pbar_val:
#             # Unpack the batch
#             tokenized_inputs, entailment_judgments = batch
#             input_ids = tokenized_inputs['input_ids'].to(device)
#             attention_mask = tokenized_inputs['attention_mask'].to(device)

#             # Forward pass
#             entailment_preds = entailment_model(input_ids, attention_mask)
#             # Compute loss
#             entailment_loss = classification_loss_fn(entailment_preds, entailment_judgments.to(device))
#             total_val_loss += entailment_loss.item()
#             # Compute evaluation metrics
#             acc_score = acc(entailment_preds.cpu(), entailment_judgments)
#             f1_score = f1(entailment_preds.cpu(), entailment_judgments)
#             # Accumulate metrics
#             total_acc += acc_score.item()
#             total_f1 += f1_score.item()

#     # Calculate average validation metrics
#     avg_val_loss = total_val_loss / len(dl_validation_entailment)
#     avg_acc = total_acc / len(dl_validation_entailment)
#     avg_f1 = total_f1 / len(dl_validation_entailment)
    
#     # Print results for the epoch
#     print(f"\nEpoch [{ep+1}/{epochs}]")
#     print(f"Entailment Judgment Training Loss: {avg_train_loss:.2f}")
#     print(f"Entailment Judgment Validation Loss: {avg_val_loss:.2f}")
#     print(f"Accuracy: {avg_acc:.2f}, F1 Score: {avg_f1:.2f}\n")

# # Testing loop for entailment judgment
# entailment_model.eval()
# total_test_loss = 0
# total_acc = 0
# total_f1 = 0
# pbar_test = tqdm(dl_test_entailment, desc=f"Testing")
# with torch.no_grad():
#     for batch in pbar_test:
#         # Unpack the batch
#         tokenized_inputs, entailment_judgments = batch
#         input_ids = tokenized_inputs['input_ids'].to(device)
#         attention_mask = tokenized_inputs['attention_mask'].to(device)

#         # Forward pass
#         entailment_preds = entailment_model(input_ids, attention_mask)
#         # Compute loss
#         entailment_loss = classification_loss_fn(entailment_preds, entailment_judgments.to(device))
#         total_test_loss += entailment_loss.item()
#         # Compute evaluation metrics
#         acc_score = acc(entailment_preds.cpu(), entailment_judgments)
#         f1_score = f1(entailment_preds.cpu(), entailment_judgments)
#         # Accumulate metrics
#         total_acc += acc_score.item()
#         total_f1 += f1_score.item()

# # Calculate average validation metrics
# avg_test_loss = total_test_loss / len(dl_test_entailment)
# avg_acc = total_acc / len(dl_test_entailment)
# avg_f1 = total_f1 / len(dl_test_entailment)

# # Print results for the epoch
# print(f"Entailment Judgment Testing Loss: {avg_test_loss:.2f}")
# print(f"Accuracy: {avg_acc:.2f}, F1 Score: {avg_f1:.2f}\n")
