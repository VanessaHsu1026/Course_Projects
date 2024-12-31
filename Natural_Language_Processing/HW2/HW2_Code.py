
# LSTM-arithmetic

import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import os

df_train = pd.read_csv(os.path.join('arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join('arithmetic_eval.csv'))
df_train.head()

# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

"""# Build Dictionary
 - The model cannot perform calculations directly with plain text.
 - Convert all text (numbers/symbols) into numerical representations.
 - Special tokens
    - '&lt;pad&gt;'
        - Each sentence within a batch may have different lengths.
        - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
    - '&lt;eos&gt;'
        - Specifies the end of the generated sequence.
        - Without '&lt;eos&gt;', the model will not know when to stop generating.
"""

char_to_id = {}
id_to_char = {}

# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

# Initialize special tokens
special_tokens = ['<pad>', '<eos>']

# Tokenize the data by extracting unique characters from the src and tgt columns
tokens = set()
for index, row in df_train.iterrows():
    for char in row['src']:
        tokens.add(char)
    for char in row['tgt']:
        tokens.add(char)

# Remove special tokens from tokens set to avoid duplicates
tokens = sorted(list(tokens - set(special_tokens)))

# Explicitly set IDs for special tokens
char_to_id = {'<pad>': 0, '<eos>': 1}

# Assign IDs to the rest of the tokens (starting from 2)
for idx, char in enumerate(tokens, start=2):
    char_to_id[char] = idx

# Create the reverse mapping (ID to char)
id_to_char = {idx: char for char, idx in char_to_id.items()}

# Print the vocabulary mappings
print('char_to_id:', char_to_id)
print('id_to_char:', id_to_char)

vocab_size = len(char_to_id)
print('Vocab size: {}'.format(vocab_size))

"""# Data Preprocessing
 - The data is processed into the format required for the model's input and output.
 - Example: 1+2-3=0
     - Model input: 1 + 2 - 3 = 0
     - Model output: / / / / / = 0 &lt;eos&gt;  (the '/' can be replaced with <pad>)
     - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate <eos>;

"""

df_train = pd.DataFrame(df_train)
df_eval = pd.DataFrame(df_eval)

# Drop the 'Unnamed: 0' column
df_train = df_train.drop(columns=['Unnamed: 0'], errors='ignore')
df_eval = df_eval.drop(columns=['Unnamed: 0'], errors='ignore')

def preprocess(row):
    src = row['src']
    tgt = row['tgt']

    # Convert each character in the source to its corresponding ID
    char_id_list = [char_to_id[char] for char in src]

    # Add the <eos> token to char_id_list
    char_id_list.append(char_to_id['<eos>'])

    # Create the label_id_list
    equal_pos = src.index('=')  # Find the position of the '='

    # Create label_id_list with <pad> tokens for the part before and including '='
    label_id_list = [char_to_id['<pad>']] * equal_pos

    # Add '=' to the output (the model should generate the '=')
    label_id_list.append(char_to_id['='])

    # Convert each character in the target to its corresponding ID
    for char in tgt:
        label_id_list.append(char_to_id[char])

    # Add the <eos> token to label_id_list
    label_id_list.append(char_to_id['<eos>'])

    # Return the processed data as a Series
    return pd.Series([char_id_list, label_id_list, len(char_id_list)])

# Apply the preprocess function to create 'char_id_list' and 'label_id_list' for both train and eval datasets
df_train[['char_id_list', 'label_id_list', 'len']] = df_train.apply(preprocess, axis=1)
df_eval[['char_id_list', 'label_id_list', 'len']] = df_eval.apply(preprocess, axis=1)

# Verify that the DataFrame is processed correctly
df_train.head()
df_eval.head()

"""# Hyper Parameters

|Hyperparameter|Meaning|Value|
|-|-|-|
|`batch_size`|Number of data samples in a single batch|64|
|`epochs`|Total number of epochs to train|3|
|`embed_dim`|Dimension of the word embeddings|256|
|`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
|`lr`|Learning Rate|0.001|
|`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|
"""

batch_size = 64
epochs = 3
embed_dim = 256
hidden_dim = 256
lr = 0.001
# lr = 0.005
# lr = 0.0002
grad_clip = 1

"""# Data Batching
- Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
- Then, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.
"""

from torch.utils.data import Dataset, DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)

    def __getitem__(self, index):
        # Extract the input data x and the ground truth y from the data
        x = self.sequences.iloc[index, 0]  # char_id_list (input sequence)
        y = self.sequences.iloc[index, 1]  # label_id_list (output sequence)
        
        x = x[:-1]  # Input: all but the last element
        y = y[1:]   # Output: all but the first element
        
        return x, y

# collate function, used to build dataloader
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                            batch_first=True,
                            padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                            batch_first=True,
                            padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])
ds_eval = Dataset(df_eval[['char_id_list', 'label_id_list']])

# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

"""# Model Design

## Execution Flow
1. Convert all characters in the sentence into embeddings.
2. Pass the embeddings through an LSTM sequentially.
3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
5. The character corresponding to the maximum value across all output dimensions is selected as the next character.

## Loss Function
Since this is a classification task, Cross Entropy is used as the loss function.

## Gradient Update
Adam algorithm is used for gradient updates.
"""
## LSTM Model
# Define the CharLSTM class
class CharLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharLSTM, self).__init__()

        # Embedding layer
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                           embedding_dim=embed_dim,
                           padding_idx=char_to_id['<pad>'])

        # First LSTM layer
        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                         hidden_size=hidden_dim,
                         batch_first=True)

        # Second LSTM layer
        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                         hidden_size=hidden_dim,
                         batch_first=True)

        # Fully connected layers
        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                  out_features=hidden_dim),
                          torch.nn.ReLU(),
                          torch.nn.Linear(in_features=hidden_dim,
                                  out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        # Convert input characters to embeddings
        batch_x = self.embedding(batch_x)

        # Pack padded sequence (since different sequences have different lengths)
        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                  batch_x_lens,
                                  batch_first=True,
                                  enforce_sorted=False)

        # Pass through the first LSTM layer
        batch_x, _ = self.rnn_layer1(batch_x)
        # Pass through the second LSTM layer
        batch_x, _ = self.rnn_layer2(batch_x)
        # Unpack the sequence
        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)
        # Pass through fully connected layers
        batch_x = self.linear(batch_x)

        return batch_x

    # Character generator for sequence generation
    def generator(self, start_char, max_len=200):
        
        char_list = [char_to_id[c] for c in start_char]
        next_char = None
        
        while len(char_list) < max_len: 
            # Convert char_list to tensor
            input_tensor = torch.tensor([char_list], dtype=torch.long).to(device)
            
            # Pass through embedding and LSTM layers
            with torch.no_grad():
                embedded = self.embedding(input_tensor)
                lstm_out, _ = self.rnn_layer1(embedded)
                lstm_out, _ = self.rnn_layer2(lstm_out)
                output = self.linear(lstm_out[:, -1, :])  # Get output from last time step
            
            # Get the next token prediction
            next_char = torch.argmax(output, dim=-1).item()
            
            if next_char == char_to_id['<eos>']:
                break
            
            char_list.append(next_char)
            
        return [id_to_char[ch_id] for ch_id in char_list]

# ## RNN Model
# # Define the CharRNN class
# class CharRNN(torch.nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super(CharRNN, self).__init__()

#         # Embedding layer
#         self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
#                                             embedding_dim=embed_dim,
#                                             padding_idx=char_to_id['<pad>'])

#         # First RNN layer
#         self.rnn_layer1 = torch.nn.RNN(input_size=embed_dim,
#                                        hidden_size=hidden_dim,
#                                        batch_first=True)

#         # Second RNN layer
#         # self.rnn_layer2 = torch.nn.RNN(input_size=hidden_dim,
#         #                                hidden_size=hidden_dim,
#         #                                batch_first=True)

#         # Fully connected layers
#         self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
#                                                           out_features=hidden_dim),
#                                           torch.nn.ReLU(),
#                                           torch.nn.Linear(in_features=hidden_dim,
#                                                           out_features=vocab_size))

#     def forward(self, batch_x, batch_x_lens):
#         return self.encoder(batch_x, batch_x_lens)

#     # The forward pass of the model
#     def encoder(self, batch_x, batch_x_lens):
#         # Convert input characters to embeddings
#         batch_x = self.embedding(batch_x)

#         # Pack padded sequence (since different sequences have different lengths)
#         batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
#                                                           batch_x_lens,
#                                                           batch_first=True,
#                                                           enforce_sorted=False)

#         # Pass through the first RNN layer
#         batch_x, _ = self.rnn_layer1(batch_x)
#         # # Pass through the second RNN layer
#         # batch_x, _ = self.rnn_layer2(batch_x)

#         # Unpack the sequence
#         batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)

#         # Pass through fully connected layers
#         batch_x = self.linear(batch_x)

#         return batch_x

#     # Character generator for sequence generation
#     def generator(self, start_char, max_len=200):
        
#         char_list = [char_to_id[c] for c in start_char]
        
#         next_char = None
        
#         while len(char_list) < max_len: 
#             # Convert char_list to tensor
#             input_tensor = torch.tensor([char_list], dtype=torch.long).to(device)
            
#             # Pass through embedding and RNN layers
#             with torch.no_grad():
#                 embedded = self.embedding(input_tensor)
#                 rnn_out, _ = self.rnn_layer1(embedded)
#                 # rnn_out, _ = self.rnn_layer2(rnn_out) 
#                 output = self.linear(rnn_out[:, -1, :])  # Get output from the last time step
            
#             # Get the next token prediction
#             next_char = torch.argmax(output, dim=-1).item()
            
#             if next_char == char_to_id['<eos>']:
#                 break
            
#             char_list.append(next_char)
            
#         return [id_to_char[ch_id] for ch_id in char_list]

# ## GRU Model
# class CharGRU(torch.nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super(CharGRU, self).__init__()

#         # Embedding layer
#         self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
#                            embedding_dim=embed_dim,
#                            padding_idx=char_to_id['<pad>'])

#         # First GRU layer
#         self.rnn_layer1 = torch.nn.GRU(input_size=embed_dim,
#                          hidden_size=hidden_dim,
#                          batch_first=True)

#         # Second GRU layer
#         self.rnn_layer2 = torch.nn.GRU(input_size=hidden_dim,
#                          hidden_size=hidden_dim,
#                          batch_first=True)

#         # Fully connected layers
#         self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
#                                   out_features=hidden_dim),
#                           torch.nn.ReLU(),
#                           torch.nn.Linear(in_features=hidden_dim,
#                                   out_features=vocab_size))

#     def forward(self, batch_x, batch_x_lens):
#         return self.encoder(batch_x, batch_x_lens)

#     # The forward pass of the model
#     def encoder(self, batch_x, batch_x_lens):
#         # Convert input characters to embeddings
#         batch_x = self.embedding(batch_x)

#         # Pack padded sequence (since different sequences have different lengths)
#         batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
#                                   batch_x_lens,
#                                   batch_first=True,
#                                   enforce_sorted=False)

#         # Pass through the first GRU layer
#         batch_x, _ = self.rnn_layer1(batch_x)
#         # Pass through the second GRU layer
#         batch_x, _ = self.rnn_layer2(batch_x)

#         # Unpack the sequence
#         batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x, batch_first=True)

#         # Pass through fully connected layers
#         batch_x = self.linear(batch_x)

#         return batch_x

#     # Character generator for sequence generation
#     def generator(self, start_char, max_len=200):

#         char_list = [char_to_id[c] for c in start_char]
#         next_char = None

#         while len(char_list) < max_len:
#             # Convert char_list to tensor
#             input_tensor = torch.tensor([char_list], dtype=torch.long).to(device)

#             # Pass through embedding and GRU layers
#             with torch.no_grad():
#                 embedded = self.embedding(input_tensor)
#                 gru_out, _ = self.rnn_layer1(embedded)
#                 gru_out, _ = self.rnn_layer2(gru_out)
#                 output = self.linear(gru_out[:, -1, :])  # Get output from the last time step

#             # Get the next token prediction
#             next_char = torch.argmax(output, dim=-1).item()

#             if next_char == char_to_id['<eos>']:
#                 break

#             char_list.append(next_char)

#         return [id_to_char[ch_id] for ch_id in char_list]

torch.manual_seed(2)

# Set device (either cuda or cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the selected device
model_LSTM = CharLSTM(vocab_size, embed_dim, hidden_dim).to(device)
# model_RNN = CharRNN(vocab_size, embed_dim, hidden_dim).to(device)
# model_GRU = CharGRU(vocab_size, embed_dim, hidden_dim).to(device)

# Define the loss function (Cross Entropy Loss, ignoring the <pad> token)
criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id['<pad>'])

# Define the optimizer (Adam)
optimizer = torch.optim.Adam(model_LSTM.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model_RNN.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model_GRU.parameters(), lr=lr)

# Model architecture
print(model_LSTM)
# print(model_RNN)
# print(model_GRU)

"""# Training
1. The outer `for` loop controls the `epoch`
    1. The inner `for` loop uses `data_loader` to retrieve batches.
        1. Pass the batch to the `model` for training.
        2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
        3. Use `loss.backward` to automatically compute the gradients.
        4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` < and < `grad_clip`.
        5. Use `optimizer.step()` to update the model (backpropagation).
2.  After every `5000` batches, output the current loss to monitor whether it is converging.
"""

# I used ChatGPT to assist me in writing code and debugging.

from tqdm import tqdm

# Lists to store loss values
train_losses = []
batch_losses = []  # List to store loss values for every 5000 batches

# Move model to the specified device
model = model_LSTM.to(device)
# model = model_RNN.to(device)
# model = model_GRU.to(device)

model.train()  # Set model to training mode
i = 0  # Initialize batch counter

for epoch in range(1, epochs + 1):
    
    # Initialize variables to accumulate losses
    total_train_loss = 0
    batch_loss = 0 
    
    # Training loop with progress bar
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass through the model
        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Reshape prediction and target tensors to calculate loss
        batch_pred_y = batch_pred_y.view(-1, vocab_size)
        batch_y = batch_y.view(-1).to(device)

        # Compute loss using Cross-Entropy Loss (ignoring <pad>)
        loss = criterion(batch_pred_y, batch_y)

        # Backpropagation: compute gradients
        loss.backward()

        # Clip the gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        # Update model parameters
        optimizer.step()
        
        # Accumulate training loss
        total_train_loss += loss.item()
        batch_loss += loss.item()
        
        i += 1
        
        # Log loss every 5000 batches
        if i % 5000 == 0:
            avg_batch_loss = batch_loss / 5000
            batch_losses.append(avg_batch_loss)
            print(f"Batch {i}, Loss: {avg_batch_loss:.4f}")

            # Reset batch loss counter
            batch_loss = 0
            
        if i % 50 == 0:
            bar.set_postfix(loss=loss.item())    
        
    # Average training loss for the epoch
    avg_train_loss = total_train_loss / len(dl_train)
    train_losses.append(avg_train_loss)
    
    # Validation loop with progress bar
    model.eval()  # Set model to evaluation mode
    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    
    matched = 0
    total = 0
    
    # predictions_all = []  # Store all predictions to print later
    # ground_truth_all = []  # Store all ground truth labels to print later
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
            
            # Forward pass through the model
            batch_pred_y = model(batch_x.to(device), batch_x_lens)
            
            # Get predicted labels by selecting the argmax across vocab dimensions
            predictions = batch_pred_y.argmax(dim=-1)

            # Flatten predictions and labels for comparison
            predictions = predictions.view(-1).cpu()
            batch_y = batch_y.view(-1).cpu()
            
            # # Convert predictions and ground truth IDs to characters
            # predictions_chars = [id_to_char[id] for id in predictions[:15].numpy()]  # Convert first 15 predictions
            # ground_truth_chars = [id_to_char[id] for id in batch_y[:15].numpy()]     # Convert first 15 ground truths

            # predictions_all.append(predictions_chars)
            # ground_truth_all.append(ground_truth_chars)

            # Compute exact match: check how many predictions match the true labels
            mask = batch_y != char_to_id['<pad>']  # Ignore pad tokens
            matched += (predictions[mask] == batch_y[mask]).sum().item()
            total += mask.sum().item()  # Total non-pad tokens

    # # Print predictions and ground truth for the last batch after validation loop
    # print("Predictions:", ''.join(predictions_all[-1]))  # Print the last batch of predictions as characters
    # print("Ground Truth:", ''.join(ground_truth_all[-1]))  # Print the last batch of predictions as characters
    
    # Compute and print exact match (EM) accuracy
    exact_match_acc = matched / total if total > 0 else 0
    print(f"Exact Match Accuracy: {exact_match_acc * 100:.2f}%")

    # Print training loss for the epoch
    print(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}")
    
    # Set the model back to training mode
    model.train()

# Plot the training loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

# Save the model's state dictionary
torch.save({
    'model_LSTM_state_dict': model.state_dict(),
    'optimizer_LSTM_state_dict': optimizer.state_dict(),
}, 'char_LSTM_model.pth')
# torch.save({
#     'model_RNN_state_dict': model.state_dict(),
#     'optimizer_RNN_state_dict': optimizer.state_dict(),
# }, 'char_RNN_model.pth')
# torch.save({
#     'model_GRU_state_dict': model.state_dict(),
#     'optimizer_GRU_state_dict': optimizer.state_dict(),
# }, 'char_GRU_model.pth')

print("LSTM Model saved successfully.")
# print("RNN Model saved successfully.")
# print("GRU Model saved successfully.")

# Load the saved state dictionary into the new model
checkpoint = torch.load('char_LSTM_model.pth')
model.load_state_dict(checkpoint['model_LSTM_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_LSTM_state_dict'])
# checkpoint = torch.load('char_RNN_model.pth')
# model.load_state_dict(checkpoint['model_RNN_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_RNN_state_dict'])
# checkpoint = torch.load('char_GRU_model.pth')
# model.load_state_dict(checkpoint['model_GRU_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_GRU_state_dict'])

print("LSTM Model loaded successfully.")
# print("RNN Model loaded successfully.")
# print("GRU Model loaded successfully.")

"""# Generation
Use `model.generator` and provide an initial character to automatically generate a sequence.
"""

model = model.to("cpu")
print("".join(model.generator('1+1='))) # 2
print("".join(model.generator('2-3='))) # -1
print("".join(model.generator('4*6='))) # 24
print("".join(model.generator('5+(7-9)='))) # 3
print("".join(model.generator('3*(10+4)='))) # 42
# train data
# one-digit number answer
print("".join(model.generator('4+(18-13)='))) # 9
print("".join(model.generator('0*(34-46)='))) # 0
# two-digit numbers answer
print("".join(model.generator('(39+25)-48='))) # 16
print("".join(model.generator('(48+49)*1='))) # 97
# three-digit numbers answer
print("".join(model.generator('42+40+22='))) # 104
print("".join(model.generator('8*(30+10)='))) # 320
# four-digit numbers answer
print("".join(model.generator('(47-23)*42='))) # 1008
print("".join(model.generator('7+32*44='))) # 1415
# eval data
# one-digit number answer
print("".join(model.generator('3+12-22='))) # -7
print("".join(model.generator('10*5-41='))) # 9
# two-digit numbers answer
print("".join(model.generator('(41-13)+25='))) # 53
print("".join(model.generator('(19*5)-29='))) # 66
# three-digit numbers answer
print("".join(model.generator('48+43+34='))) # 125
print("".join(model.generator('9*(24-11)='))) # 117
# four-digit numbers answer
print("".join(model.generator('11-36*29='))) # -1033
print("".join(model.generator('8+40*37='))) # 1488

# Some numbers [51, 100] never appear in training data
print("".join(model.generator('65+80='))) # 145
print("".join(model.generator('72-53='))) # 19
print("".join(model.generator('91*100='))) # 9100
print("".join(model.generator('64*(81-52)='))) # 1856
print("".join(model.generator('77*(59+43)='))) # 7854