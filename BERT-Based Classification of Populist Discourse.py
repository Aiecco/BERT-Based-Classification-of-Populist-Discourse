#%%
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



n_epochs = 50
l_rate = 3e-5
w_decay = 0.01



df = pd.read_excel(r"C:\Users\alepa\Desktop\Database_NLP_Alessandro.xlsx", header=None)

labels = df.iloc[0, 0::2].tolist()
speeches = df.iloc[0, 1::2].tolist()

labels = [int(label) for label in labels]

data = pd.DataFrame({'label': labels, 'speech': speeches})

train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['label'])

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



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = SpeechDataset(train_data['speech'].tolist(), train_data['label'].tolist(), tokenizer)
test_dataset = SpeechDataset(test_data['speech'].tolist(), test_data['label'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.train()
model.to(device)



optimizer = torch.optim.AdamW(model.parameters(),
                              lr = l_rate, 
                              weight_decay = w_decay)



def live_plot(train_data, test_data, figsize=(10, 5), title=''):
    clear_output(wait=True)
    
    df_train = pd.DataFrame(train_data)
    df_train['Type'] = 'Train'
    df_test = pd.DataFrame(test_data)
    df_test['Type'] = 'Test'
    df = pd.concat([df_train, df_test], axis=0)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Epoch'}, inplace=True)

    sns.set(style='dark')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.lineplot(x='Epoch', y='Loss', hue='Type',
                 data=df.melt(id_vars=['Epoch', 'Type'], value_vars=['Train Loss', 'Test Loss'], var_name='Metric',
                              value_name='Loss'), ax=ax1)
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')

    sns.lineplot(x='Epoch', y='Accuracy', hue='Type',
                 data=df.melt(id_vars=['Epoch', 'Type'], value_vars=['Train Accuracy', 'Test Accuracy'],
                              var_name='Metric', value_name='Accuracy'), ax=ax2)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='lower right')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

train_metrics = {'Train Loss': [], 'Train Accuracy': []}
test_metrics = {'Test Loss': [], 'Test Accuracy': []}

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []




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
