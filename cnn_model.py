import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ===== Directory Setup =====
t = datetime.fromtimestamp(time.time())
save_dir_name = 'capa_16x16'
save_dir_name = save_dir_name + '_' + datetime.strftime(t, '%m%d')

CODES_DIR = os.path.dirname(os.getcwd())
DATAROOT = os.path.join(CODES_DIR, 'processed')
SAVE_PATH = 'results'
save_dir_name = os.path.join(SAVE_PATH, save_dir_name)

for p in [DATAROOT, SAVE_PATH, save_dir_name]:
    if not os.path.exists(p):
        os.mkdir(p)

# ===== Parameters =====
num_epoch = 100
learning_rate = 0.01

# ===== Load Dataset =====
feature = []

for n in range(50):
    i = n + 1
    feature_data = np.load(f'capa_processed_data50/{i}.npy', allow_pickle=True)
    for j in range(252):
        feature.append(feature_data[j])

np.random.shuffle(feature)

dataset = feature

# Split dataset
tr_data = np.array([d[0] for d in dataset[0:8000]])
tr_label = np.array([d[1] for d in dataset[0:8000]])

te_data = np.array([d[0] for d in dataset[8000:]])
te_label = np.array([d[1] for d in dataset[8000:]])

# Convert to torch tensors
tr_data = torch.tensor(np.expand_dims(tr_data, 1), dtype=torch.float)
tr_label = torch.tensor(tr_label, dtype=torch.long)

te_data = torch.tensor(np.expand_dims(te_data, 1), dtype=torch.float)
te_label = torch.tensor(te_label, dtype=torch.long)

# ===== Model Definition =====
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1, bias=False)

        self.fc1 = nn.Linear(128, 100)
        self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# ===== Training Setup =====
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# ===== Training Loop =====
start_time = time.time()

loss_list = []
test_acc_list = []

for epoch in range(num_epoch):

    optimizer.zero_grad()

    output = model(tr_data)
    loss = criterion(output, tr_label)

    pred = output.argmax(dim=1)
    train_acc = (pred == tr_label).sum().item() / len(tr_data)

    loss.backward()
    optimizer.step()
    scheduler.step()

    loss_list.append(loss.item())

    print(f"Epoch {epoch}: Loss = {loss:.4f}, Train Acc = {train_acc:.6f}")

    # ===== Evaluation =====
    with torch.no_grad():
        test_output = model(te_data)
        test_pred = test_output.argmax(dim=1)
        test_acc = (test_pred == te_label).sum().item() / len(te_data)

        test_acc_list.append(test_acc)

        print(f"Test Accuracy: {test_acc:.6f}")

# ===== Confusion Matrix =====
conf_mat = confusion_matrix(te_label, test_pred)

conf_mat_df = pd.DataFrame(conf_mat)
conf_mat_norm = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0)

torch.save({'conf_mat': conf_mat_df}, os.path.join(save_dir_name, 'conf_mat.pt'))
torch.save({'conf_mat_norm': conf_mat_norm}, os.path.join(save_dir_name, 'conf_mat_norm.pt'))

# ===== Visualization =====
plt.figure(figsize=(20, 16))
sns.heatmap(conf_mat_df, annot=True, fmt='d')
plt.savefig(os.path.join(save_dir_name, 'conf_mat.png'))
plt.close()

plt.figure(figsize=(20, 16))
sns.heatmap(conf_mat_norm, annot=True, fmt='.2f')
plt.savefig(os.path.join(save_dir_name, 'conf_mat_normalized.png'))
plt.close()

plt.figure()
plt.plot(range(num_epoch), test_acc_list, '+r-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Epoch')
plt.savefig(os.path.join(save_dir_name, 'accuracy_curve.png'))
plt.close()