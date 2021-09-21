import requests
from PIL import Image
from azure.storage.blob import BlobClient
import folium
import numpy as np
import datetime
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_test_validate_split(df, proportions, part_colname = "partition"):
    """
    Takes a DataFrame (`df`) and splits it into train, test, and validate
    partitions. Returns a DataFrame with a new column, `part_colname` specifying
    which partition each row belongs to. `proportions` is a list of length 3 with 
    desired proportions for train, test, and validate partitions, in that order.
    """
    if sum(proportions) != 1 | len(proportions) != 3:
        sys.exit("Error: proportions must be length 3 and sum to 1.")
    
    # first sample train data
    train = df.sample(frac=proportions[0], random_state=2)
    train[part_colname] = "train"
    # drop train data from the df
    test_validate = df.drop(train.index)
    # sample test data
    test = test_validate.sample(frac=proportions[1]/sum(proportions[1:3]), random_state=2)
    test[part_colname] = "test"
    #drop test data from test_validate, leaving you with validate in correct propotion
    validate = test_validate.drop(test.index)
    validate[part_colname] = "validate"

    return pd.concat([train, test, validate])

def dates_to_julian(stddate):
    fmt='%Y-%m-%d'
    sdtdate = datetime.datetime.strptime(stddate, fmt)
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    return(jdate)
                
def url_to_img(url):
    response = requests.get(url, stream=True)
    response.raw.decode_content = True
    image = Image.open(response.raw)
    return image

def local_to_blob(container, localfile, blobname, storage_options):
    account_url = f"https://{storage_options['account_name']}.blob.core.windows.net"
    blobclient = BlobClient(account_url=account_url,\
                container_name=container,\
                blob_name=blobname,\
                credential=storage_options['account_key'])
    with open(f"{localfile}", "rb") as out_blob:
        blob_data = blobclient.upload_blob(out_blob, overwrite=True)

def generate_map(df, lat_colname='Latitude', lon_colname='Longitude'):
    '''
    plots web map using folium
    '''
    cx, cy = np.mean(df[lon_colname]),np.mean(df[lat_colname])
    plot_map = folium.Map(location=[cy, cx],\
                zoom_start=2,\
                tiles='CartoDB positron')
    for _, r in df.iterrows():
        folium.Marker(location=[r[lat_colname], r[lon_colname]],\
            #popup=f"{r['site_no']}:\n{r['station_name']}").add_to(self.plot_map)
            popup=f"Site: {r['site_no']}").add_to(plot_map)
        #polygons = gpd.GeoSeries(r['buffer_geometry'])
        #geo_j = polygons.to_json()
        #geo_j = folium.GeoJson(data=geo_j,style_function=lambda x: {'fillColor': 'orange'})
        #geo_j.add_to(plot_map)
    tile = folium.TileLayer(\
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\
                attr = 'Esri',\
                name = 'Esri Satellite',\
                overlay = False,\
                control = True).add_to(plot_map)

    return plot_map


def fit_mlp(
        features,
        learning_rate,
        batch_size,
        epochs,
        storage_options,
        buffer_distance=500,
        day_tolerance=8,
        cloud_thr=80,
        mask_method="lulc",
        min_water_pixels=1
    ):
    fp = f"az://modeling-data/partitioned_feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mask_method}_masking.csv"
    data = pd.read_csv(fp, storage_options=storage_options)

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

    # Train the model
    print("Begin training.")
    for e in range(1, epochs+1):
        # TRAINING
        train_epoch_loss = 0
        model.train()

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

        if (e % 10 == 0):
            print(f"Epoch {e}/{epochs} | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}")

    # plt.figure(figsize=(8,6))
    # plt.plot(loss_stats["train"], color="teal", label="training")
    # plt.plot(loss_stats["val"], color="orange", label="validation")
    # plt.title("Loss Curves for Training and Validation Data")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")

    # plt.legend()
    # plt.savefig("/content/figs/MLP_prelim_results/loss_curves.png", bbox_inches="tight", facecolor="#FFFFFF", dpi=150)

    test_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            test_pred_list.append(y_pred.cpu().numpy())
    test_pred_list = [a.squeeze().tolist() for a in test_pred_list]
    test_mse = mean_squared_error(test_pred_list, y_test)
    test_r_squared = r2_score(test_pred_list, y_test)

    train_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in train_loader_all:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            train_pred_list.append(y_pred.cpu().numpy())
    train_pred_list = [a.squeeze().tolist() for a in train_pred_list]
    train_mse = mean_squared_error(train_pred_list, y_train)
    train_r_squared = r2_score(train_pred_list, y_train)
    
    itv_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in itv_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            itv_pred_list.append(y_pred.cpu().numpy())
    itv_pred_list = [a.squeeze().tolist() for a in itv_pred_list]
    itv_mse = mean_squared_error(itv_pred_list, y_itv)
    itv_r_squared = r2_score(itv_pred_list, y_itv)

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
        "test_obs_predict": pd.DataFrame({
            "Test set predictions": test_pred_list,
            "Test set observations": y_test
        }),
        "test_mse": test_mse,
        "test_R2": test_r_squared,
        "train_obs_predict": pd.DataFrame({
            "Train set predictions": train_pred_list,
            "Train set observations": y_train
        }),
        "train_mse": train_mse,
        "train_R2": train_r_squared,
        "itv_obs_predict": pd.DataFrame({
            "Test set predictions": itv_pred_list,
            "Test set observations": y_itv,
        }),
        "itv_mse": itv_mse,
        "itv_R2": itv_r_squared
    }
    
    return output

def plot_obs_predict(obs_pred, title, savefig=False, outfn=""):
    plt.figure(figsize=(8,8))
    plt.plot(list(range(1,8)),list(range(1,8)), color="black", label="One-to-one 1 line")
    plt.scatter(obs_pred.iloc[:,0], obs_pred.iloc[:,1])
    plt.xlabel("ln(SSC) Predicted")
    plt.ylabel("ln(SSC) Observed")
    plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig(
            outfn,
            bbox_inches="tight",
            facecolor="#FFFFFF",
            dpi=150
        )