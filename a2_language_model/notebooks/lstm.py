import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import datasets
import math
from tqdm import tqdm
import time
from lstm_model import LSTMLanguageModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load dataset and tokenizer
dataset = datasets.load_dataset('KaungHtetCho/Harry_Potter_LSTM')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Preprocess dataset
filtered_dataset = dataset.filter(lambda x: x['text'].strip() != '')
tokenize_data = lambda x: {'tokens': tokenizer(x['text'])}
tokenized_dataset = filtered_dataset.map(tokenize_data, remove_columns=[], fn_kwargs={'tokenizer': tokenizer})
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['tokens']) > 0)

# Build vocabulary
vocab = torchtext.vocab.build_vocab_from_iterator(
    tokenized_dataset['train']['tokens'], 
    min_freq=3, 
    specials=['<unk>', '<eos>']
)
vocab.set_default_index(vocab['<unk>'])

# Function to prepare data for batching
def get_data(dataset, vocab, batch_size):
    data = []
    for example in dataset:
        tokens = example['tokens'] + ['<eos>']
        indices = [vocab[token] for token in tokens]
        data.extend(indices)
    data = torch.tensor(data, dtype=torch.long)
    num_batches = data.size(0) // batch_size
    data = data[:num_batches * batch_size].view(batch_size, -1)
    return data

# Prepare train, validation, and test data
batch_size = 128
train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
valid_data = get_data(tokenized_dataset['validation'], vocab, batch_size)
test_data = get_data(tokenized_dataset['test'], vocab, batch_size)

# Model configuration
vocab_size = len(vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65
lr = 1e-3

# Initialize model, optimizer, and loss function
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training and evaluation functions
def get_batch(data, seq_len, idx):
    src = data[:, idx:idx + seq_len]
    target = data[:, idx + 1:idx + seq_len + 1]
    return src, target

def train_epoch(model, data, optimizer, criterion, seq_len, clip, device):
    model.train()
    epoch_loss = 0
    hidden = model.init_hidden(data.size(0), device)

    for idx in tqdm(range(0, data.size(1) - seq_len, seq_len), desc="Training", leave=False):
        src, target = get_batch(data, seq_len, idx)
        src, target = src.to(device), target.to(device)

        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        prediction, hidden = model(src, hidden)

        prediction = prediction.view(-1, vocab_size)
        target = target.view(-1)
        loss = criterion(prediction, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / (data.size(1) // seq_len)

def evaluate(model, data, criterion, seq_len, device):
    model.eval()
    epoch_loss = 0
    hidden = model.init_hidden(data.size(0), device)

    with torch.no_grad():
        for idx in range(0, data.size(1) - seq_len, seq_len):
            src, target = get_batch(data, seq_len, idx)
            src, target = src.to(device), target.to(device)

            hidden = model.detach_hidden(hidden)
            prediction, hidden = model(src, hidden)

            prediction = prediction.view(-1, vocab_size)
            target = target.view(-1)
            loss = criterion(prediction, target)
            epoch_loss += loss.item()

    return epoch_loss / (data.size(1) // seq_len)

# Training loop
n_epochs = 50
seq_len = 50
clip = 0.25
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
best_valid_loss = float('inf')

for epoch in range(n_epochs):
    train_loss = train_epoch(model, train_data, optimizer, criterion, seq_len, clip, device)
    valid_loss = evaluate(model, valid_data, criterion, seq_len, device)

    lr_scheduler.step(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "best_model_lstm.pt")

    print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}")

# Test evaluation
model.load_state_dict(torch.load("best_model_lstm.pt"))
test_loss = evaluate(model, test_data, criterion, seq_len, device)
print(f"Test Loss: {test_loss:.3f}, Test Perplexity: {math.exp(test_loss):.3f}")

# Text generation
def generate_text(prompt, max_seq_len, temperature, model, tokenizer, vocab, device):
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[token] for token in tokens]
    hidden = model.init_hidden(1, device)

    with torch.no_grad():
        for _ in range(max_seq_len):
            src = torch.tensor([indices], dtype=torch.long).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == vocab["<eos>"]:
                break
            indices.append(next_token)

    itos = vocab.get_itos()
    return " ".join(itos[idx] for idx in indices)

# Generate text with various temperatures
prompt = "Harry Potter is"
for temp in [0.5, 1.0, 1.5, 2.0, 3.1415]:
    print(f"Temperature {temp}:\n{generate_text(prompt, 100, temp, model, tokenizer, vocab, device)}\n")
