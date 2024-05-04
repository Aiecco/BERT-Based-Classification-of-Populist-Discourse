#%%
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
n_epochs = 50
l_rate = 3e-5
w_decay = 0.01
#%%
df = pd.read_excel(r"C:\Users\alepa\Desktop\Database_NLP_Alessandro.xlsx", header=None)

labels = df.iloc[0, 0::2].tolist()
speeches = df.iloc[0, 1::2].tolist()

labels = [int(label) for label in labels]

data = pd.DataFrame({'label': labels, 'speech': speeches})
#%%
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'])
#%%
class SpeechDataset(Dataset):
    def __init__(self, speeches, labels, tokenizer, max_len=512):
        self.speeches = speeches
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, item):
        speech = str(self.speeches[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            speech,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = SpeechDataset(train_data['speech'].tolist(), train_data['label'].tolist(), tokenizer)
test_dataset = SpeechDataset(test_data['speech'].tolist(), test_data['label'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.train()
model.to(device)
#%%
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = l_rate, 
                              weight_decay = w_decay)
#%%
def live_plot(train_data, test_data, figsize=(7,5), title=''):
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(train_data['Train Loss'], label='Train Loss')
    ax1.plot(train_data['Train Accuracy'], label='Train Accuracy')
    ax1.set_title('Training Loss and Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(test_data['Test Loss'], label='Test Loss')
    ax2.plot(test_data['Test Accuracy'], label='Test Accuracy')
    ax2.set_title('Testing Loss and Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.suptitle(title)
    plt.show()

train_metrics = {'Train Loss': [], 'Train Accuracy': []}
test_metrics = {'Test Loss': [], 'Test Accuracy': []}

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []
#%%
for epoch in range(n_epochs):
    model.train()
    total_train_loss, total_train_accuracy = 0, 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_accuracy += (outputs.logits.argmax(dim=-1) == batch['labels']).float().mean().item()

    train_metrics['Train Loss'].append(total_train_loss / len(train_loader))
    train_metrics['Train Accuracy'].append(total_train_accuracy / len(train_loader))

    model.eval()
    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            total_test_loss += loss.item()
            total_test_accuracy += (outputs.logits.argmax(dim=-1) == batch['labels']).float().mean().item()

    test_metrics['Test Loss'].append(total_test_loss / len(test_loader))
    test_metrics['Test Accuracy'].append(total_test_accuracy / len(test_loader))

    live_plot(train_metrics, test_metrics, title='Real-time Training and Testing Metrics')