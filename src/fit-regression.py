# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.lines as mlines

# %%  Get Azure blob strorage info for pandas storage settings


with open("/content/credentials") as f:
    env_vars = f.read().split("\n")

for var in env_vars:
    key, value = var.split(" = ")
    os.environ[key] = value

storage_options = {"account_name":os.environ["ACCOUNT_NAME"],
                   "account_key":os.environ["BLOB_KEY"]}

# %% define arguments
buffer_distance = 500
day_tolerance = 8
cloud_thr = 80
min_water_pixels = 10
features = [
    "Intercept", "sentinel-2-l2a_AOT", "sentinel-2-l2a_B02",
    "sentinel-2-l2a_B03", "sentinel-2-l2a_B04", "sentinel-2-l2a_B08",
    "sentinel-2-l2a_WVP", "sentinel-2-l2a_B05", "sentinel-2-l2a_B06",
    "sentinel-2-l2a_B07", "sentinel-2-l2a_B8A", "sentinel-2-l2a_B11",
    "mean_viewing_azimuth", "mean_viewing_zenith",
    "mean_solar_azimuth", "mean_solar_zenith"
]
epochs = 700
batch_size = 128
learning_rate = 0.01


# %% Read in data, prep inputs
fp = f"az://modeling-data/partitioned_training_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent.csv"
data = pd.read_csv(fp, storage_options=storage_options)

# data = data[(data["data_src"] == "itv") | (data["data_src"] == "ana")]
not_enough_water = data["n_water_pixels"] <= min_water_pixels
data.drop(not_enough_water[not_enough_water].index, inplace=True)
data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])

lnssc_0 = data["Log SSC (mg/L)"] == 0
data.drop(lnssc_0[lnssc_0].index, inplace=True)

data["Intercept"] = 1

train = data[data["partition"] == "train"]
test = data[data["partition"] == "test"]
validate = data[data["partition"] == "validate"]
itv = data[data["data_src"] == "itv"]
response = "Log SSC (mg/L)"

y_train = train[response]
X_train = train[features]
y_test = test[response]
X_test = test[features]
y_val = validate[response]
X_val = validate[features]
y_itv = itv[response]
X_itv = itv[features]


## Subsequent code adapted from https://towardsdatascience.com/pytorch-tabular-regression-428e9c9ac93
# %% Scale the X data, prep model class
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_itv = scaler.transform(X_itv)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)
X_itv, y_itv = np.array(X_itv), np.array(y_itv)

class RegressionDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = RegressionDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
itv_dataset = RegressionDataset(torch.from_numpy(X_itv).float(), torch.from_numpy(y_itv).float())
num_features = X_test.shape[1]

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader_all = DataLoader(dataset=train_dataset, batch_size=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
itv_loader = DataLoader(dataset=itv_dataset, batch_size=1)
class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 24)
        self.layer_2 = nn.Linear(24, 48)
        self.layer_3 = nn.Linear(48, 12)
        self.layer_4 = nn.Linear(12, 6)
        self.layer_out = nn.Linear(6, 1)
        
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.layer_out(x)
        return (x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MultipleRegression(num_features)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.2)

loss_stats = {
    "train": [],
    "val": []
}

# %% Train the model
print("Begin training.")
for e in range(1, epochs+1):

    # TRAINING
    train_epoch_loss = 0
    model.train()
    # for X_train_batch, y_train_batch in train_loader:
    #     # grab data to iteration and send to CPU
    #     X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

    #     # Forward pass
    #     y_train_pred = model(X_train_batch)
        
    #     # Compute loss
    #     train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        
    #     # Zero the gradient, backward pass, update optimizer weights
    #     optimizer.zero_grad()
    #     train_loss.backward()
    #     optimizer.step()
        
    #     # Update running loss
    #     train_epoch_loss += train_loss.item()
    
    for X_train_batch, y_train_batch in train_loader:
        # grab data to iteration and send to CPU
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

        def closure():
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            y_train_pred = model(X_train_batch)
            # Compute loss
            train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
            # Backward pass
            train_loss.backward()

            return train_loss
        
        # Update weights
        optimizer.step(closure)

        # Update the running loss
        train_loss = closure()
        train_epoch_loss += train_loss.item()

    # VALIDATION    
    with torch.no_grad():
        val_epoch_loss = 0
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            
            val_epoch_loss += val_loss.item()
    loss_stats["train"].append(train_epoch_loss/len(train_loader))
    loss_stats["val"].append(val_epoch_loss/len(val_loader))                              

    scheduler.step()

    if (e % 5 == 0):
        print(f"Epoch {e}/{epochs} | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}")

# %%
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=["index"]).rename(columns={"index":"epochs"})
plt.figure(figsize=(8,6))
plt.plot(loss_stats["train"], color="teal", label="training")
plt.plot(loss_stats["val"], color="orange", label="validation")
teal_line = mlines.Line2D([], [], color="teal", label="Training")
orange_line = mlines.Line2D([], [], color="orange", label="Validation")
plt.title("Loss Curves for Training and Validation Data")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.savefig("/content/figs/MLP_prelim_results/loss_curves.png", bbox_inches="tight", facecolor="#FFFFFF", dpi=150)

# %% test the model
y_pred_list = []
compare_to = y_test
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        y_pred_list.append(y_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
mse = mean_squared_error(compare_to, y_pred_list)
r_squared = r2_score(compare_to, y_pred_list)
print("Mean Squared Error :", mse)
print("R^2 :", r_squared)

plt.figure(figsize=(8,8))
plt.plot(list(range(1,8)),list(range(1,8)), color="black", label="One-to-one 1 line")

plt.scatter(compare_to, y_pred_list)
plt.xlabel("ln(SSC) Observed")
plt.ylabel("ln(SSC) Predicted")
plt.title("Observed Vs. Predicted SSC for Test Data")
plt.legend()
plt.savefig("/content/figs/MLP_prelim_results/obs_predict_test.png", bbox_inches="tight", facecolor="#FFFFFF", dpi=150)

output = {
    "buffer_distance": buffer_distance,
    "day_tolerance": day_tolerance,
    "cloud_thr": cloud_thr,
    "min_water_pixels": min_water_pixels,
    "features": features,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "loss_stats": loss_stats,
    "epochs": epochs,
    "mse": mse,
    "R2": r_squared
}

# %%
