import requests
from PIL import Image
from azure.storage.blob import BlobClient
import folium
import numpy as np
import datetime
import pandas as pd
import sys
import fsspec
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter as gf

class MultipleRegression(nn.Module):
        def __init__(self, num_features, n_layers, layer_out_neurons, activation_function):
            super(MultipleRegression, self).__init__()
            self.n_layers = n_layers
            self.layer_out_neurons = layer_out_neurons

            most_recent_n_neurons = layer_out_neurons[0]
            self.layer_1 = nn.Linear(num_features, layer_out_neurons[0])

            for i in range(2, n_layers + 1):
                setattr(
                    self,
                    f"layer_{i}",
                    nn.Linear(layer_out_neurons[i-2], layer_out_neurons[i-1])
                )
                most_recent_n_neurons = layer_out_neurons[i-1]

            self.layer_out = nn.Linear(most_recent_n_neurons, 1)
            # self.activate = torch.nn.PReLU(num_parameters=1, init=0.1)
            self.activate = activation_function


        def forward(self, inputs):
            x = self.activate(self.layer_1(inputs))
            for i in range(2, self.n_layers + 1):
                x = self.activate(getattr(self, f"layer_{i}")(x))
            x = self.layer_out(x)

            return (x)

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


def fit_mlp_cv(
        features,
        learning_rate,
        batch_size,
        epochs,
        storage_options,
        activation_function=nn.SELU(),
        buffer_distance=500,
        day_tolerance=8,
        cloud_thr=80,
        mask_method1="lulc",
        mask_method2="mndwi",
        min_water_pixels=10,
        layer_out_neurons=[24, 12, 6],
        learn_sched_step_size=200, 
        learn_sched_gamma=0.2,
        verbose=True
    ):    
    n_layers = len(layer_out_neurons)
    torch.set_num_threads(1)
    # Read the data
    if mask_method2 == "ndvi":
        fp = f"/content/local/partitioned_feature_data_buffer500m_daytol8_cloudthr80percent_lulcndvi_masking_12folds.csv"
    elif mask_method2 == "mndwi":
         fp = f"/content/local/partitioned_feature_data_buffer500m_daytol8_cloudthr80percent_lulcmndwi_masking_tmp.csv"

    data = pd.read_csv(fp)

    data = data[data["partition"] != "testing"]
    data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])
    response = "Log SSC (mg/L)"
    not_enough_water = data["n_water_pixels"] < min_water_pixels
    data.drop(not_enough_water[not_enough_water].index, inplace=True)
    lnssc_0 = data["Log SSC (mg/L)"] == 0
    data.drop(lnssc_0[lnssc_0].index, inplace=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[features])

    fold_n_sites = []

    val_pred_fold = []
    train_pred_fold = []
    val_loss_fold = []
    val_pooled_loss_fold = []
    train_loss_fold = []
    y_val_fold = []
    y_train_fold = []
    val_site_mse = []
    val_pooled_mse = []
    val_R2_fold = []

    for fold in [i for i in range(1, len(np.unique(data["fold_idx"])) + 1)]:
        X_train = X_scaled[data["fold_idx"] != fold]
        y_train = data[data["fold_idx"] != fold][response]
        X_val = X_scaled[data["fold_idx"] == fold]
        y_val = data[data["fold_idx"] == fold][response]
        site_val = list(data[data["fold_idx"] == fold]["site_no"])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)

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

        num_features = X_train.shape[1]

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_all = DataLoader(dataset=train_dataset, batch_size=1)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = MultipleRegression(num_features, n_layers, layer_out_neurons, activation_function)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=learn_sched_step_size,
            gamma=learn_sched_gamma
        )

        loss_stats = {
            "train": [],
            "val": [],
            "val_pooled": []
        }

        val_R2 = []

        # Train the model
        print(f"Training on fold {fold}.")
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

            val_pred = [] 
            
            with torch.no_grad():
                model.eval()
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    y_pred = model(X_batch).cpu().squeeze().tolist()#.numpy()
                    val_pred.append(y_pred)

            val_pred = np.array(val_pred)
            val_se = list((val_pred - y_val)**2)

            group_means = [np.mean(val_se[site_val == a]) for a in np.unique(site_val)]
            val_loss_site = np.mean(group_means)
            val_loss = np.mean(val_se)
            loss_stats["val"].append(val_loss_site)
            loss_stats["val_pooled"].append(val_loss)
            loss_stats["train"].append(train_epoch_loss/len(train_loader))
            
            scheduler.step()

            # calculate R^2 for this epoch
            val_R2.append(r2_score(val_pred, y_val))

            if (e % 50 == 0) and verbose:
                print(f"Epoch {e}/{epochs} | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss (mean of sites): {val_loss_site:.5f} | Val Loss (pooled mean): {val_loss:.5f}")
        
        train_pred_list = []

        with torch.no_grad():
            model.eval()
            for X_batch, _ in train_loader_all:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch).cpu().squeeze().tolist()
                train_pred_list.append(y_pred)

        val_pred_list = []

        with torch.no_grad():
            model.eval()
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch).cpu().squeeze().tolist()
                val_pred_list.append(y_pred)

        val_pred = np.array(val_pred_list)
        val_se = list((val_pred - y_val)**2)
        group_means = [np.mean(val_se[site_val == a]) for a in np.unique(site_val)]

        # Overall MSE for validation (pooled, and grouped by site with equal weight)
        val_site_mse.append(np.mean(group_means))
        val_pooled_mse.append(np.mean(val_se))

        # Number of sites in each fold (used for weighted averages of MSE)
        fold_n_sites.append(len(np.unique(site_val)))

        # Per-epoch losses
        val_loss_fold.append(loss_stats["val"])
        val_pooled_loss_fold.append(loss_stats["val_pooled"])
        train_loss_fold.append(loss_stats["train"])
        
        # Observations and predictions for validation and train (for each fold)
        val_pred_fold.append(val_pred_list)
        y_val_fold.append(list(y_val))
        train_pred_fold.append(train_pred_list)
        y_train_fold.append(list(y_train))

        # Track validation R^2 per epoch per fold
        val_R2_fold.append(val_R2)


    output = {
        "training_data": fp,
        "buffer_distance": buffer_distance,
        "day_tolerance": day_tolerance,
        "cloud_thr": cloud_thr,
        "min_water_pixels": min_water_pixels,
        "features": features,
        "learning_rate": learning_rate,
        "learn_sched_step_size": learn_sched_step_size,
        "learn_sched_gamma": learn_sched_gamma,
        "batch_size": batch_size,
        "layer_out_neurons": layer_out_neurons,
        "epochs": epochs,
        "activation": f"{activation_function}",
        "train_loss_fold": train_loss_fold,
        "val_site_loss_fold": val_loss_fold,
        "val_pooled_loss_fold": val_pooled_loss_fold,
        "val_R2_fold": val_R2_fold,
        "val_site_mse": np.average(val_site_mse, weights=fold_n_sites),
        "val_pooled_mse": np.average(val_pooled_mse, weights=fold_n_sites),
        "y_obs_val_fold": y_val_fold,
        "y_pred_val_fold": val_pred_fold,
        "y_obs_train_fold": y_train_fold,
        "y_pred_train_fold": train_pred_fold
    }

    return output


def fit_mlp_full(
        features,
        learning_rate,
        batch_size,
        epochs,
        storage_options,
        activation_function=nn.SELU(),
        buffer_distance=500,
        day_tolerance=8,
        cloud_thr=80,
        mask_method1="lulc",
        mask_method2="mndwi",
        min_water_pixels=10,
        layer_out_neurons=[24, 12, 6],
        learn_sched_step_size=200, 
        learn_sched_gamma=0.2,
        verbose=True,
        model_out = "output/top_model"
    ):    
    n_layers = len(layer_out_neurons)

    # Read the data
    if mask_method2 == "ndvi":
        fp = f"/content/local/partitioned_feature_data_buffer500m_daytol8_cloudthr80percent_lulcndvi_masking_12folds.csv"
    elif mask_method2 == "mndwi":
         fp = f"/content/local/partitioned_feature_data_buffer500m_daytol8_cloudthr80percent_lulcmndwi_masking_tmp.csv"

    data = pd.read_csv(fp)
    data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])

    test = data[data["partition"] == "testing"]
    data = data[data["partition"] != "testing"]
    
    response = "Log SSC (mg/L)"
    not_enough_water = data["n_water_pixels"] < min_water_pixels
    data.drop(not_enough_water[not_enough_water].index, inplace=True)
    lnssc_0 = data["Log SSC (mg/L)"] == 0
    data.drop(lnssc_0[lnssc_0].index, inplace=True)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(data[features])
    X_test_scaled = scaler.transform(test[features])

    y_train = data[response]
    y_test = test[response]

    X_train_scaled, y_train = np.array(X_train_scaled), np.array(y_train)
    X_test_scaled, y_test = np.array(X_test_scaled), np.array(y_test)

    class RegressionDataset(Dataset):

        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]

        def __len__ (self):
            return len(self.X_data)

    train_dataset = RegressionDataset(torch.from_numpy(X_train_scaled).float(), torch.from_numpy(y_train).float())
    test_dataset = RegressionDataset(torch.from_numpy(X_test_scaled).float(), torch.from_numpy(y_test).float())

    num_features = X_train_scaled.shape[1]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_all = DataLoader(dataset=train_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MultipleRegression(num_features, n_layers, layer_out_neurons, activation_function)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=learn_sched_step_size,
        gamma=learn_sched_gamma
    )

    train_loss_list = []

    # Train the model
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
        
        train_loss_list.append(train_epoch_loss/len(train_loader))
        scheduler.step()

        if (e % 50 == 0) and verbose:
            print(f"Epoch {e}/{epochs} | Train Loss: {train_epoch_loss/len(train_loader):.5f}", end="\r")
    
    train_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in train_loader_all:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().squeeze().tolist()
            train_pred_list.append(y_pred)

    test_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            test_pred = model(X_batch).cpu().squeeze().tolist()
            test_pred_list.append(test_pred)

    test_pred = np.array(test_pred_list)
    test_se = list((test_pred - y_test)**2)

    site_test = list(test["site_no"])

    group_means = [np.mean(test_se[site_test == a]) for a in np.unique(site_test)]
    test_site_mse = np.mean(group_means)
    test_pooled_mse = np.mean(test_se)

    output = {
        "training_data": fp,
        "buffer_distance": buffer_distance,
        "day_tolerance": day_tolerance,
        "cloud_thr": cloud_thr,
        "min_water_pixels": min_water_pixels,
        "features": features,
        "learning_rate": learning_rate,
        "learn_sched_step_size": learn_sched_step_size,
        "learn_sched_gamma": learn_sched_gamma,
        "batch_size": batch_size,
        "layer_out_neurons": layer_out_neurons,
        "epochs": epochs,
        "activation": f"{activation_function}",
        "train_loss": train_loss_list,
        "train_pooled_mse": train_loss_list[-1],
        "test_site_mse": test_site_mse,
        "test_pooled_mse": test_pooled_mse,
        "y_train_sample_id": list(data["sample_id"]),
        "y_test_sample_id": list(test["sample_id"]),
        #"X_test_scaled": X_test_scaled,
        "y_obs_train": y_train,
        "y_pred_train": train_pred_list,
        "y_obs_test": y_test,
        "y_pred_test": test_pred_list
    }

    # save the model!
    torch.save(model.state_dict(), f"output/{model_out}.pt")

    with open(f"output/{model_out}_metadata.pickle", 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    fs = fsspec.filesystem("az", **storage_options)
    fs.put_file(f"output/{model_out}.pt", f"model-output/{model_out}.pt", overwrite=True)
    fs.put_file(f"/content/output/{model_out}_metadata.pickle", f"model-output/{model_out}_metadata.pickle", overwrite=True)
    
    return output


def plot_obs_predict(obs_pred, title, savefig=False, outfn=""):
    plt.figure(figsize=(8,8))
    plt.plot(list(range(0,8)),list(range(0,8)), color="black", label="One-to-one 1 line")
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


def denoise(image, operation = "erosion", kernel_size = 3, iterations = 1):

    """
    Morphological operations

    Keyword arguments:
    image -- the image
    operation -- the morphological operators (default erosion)
    kernel_size -- the size of the matrix with which image is convolved (default 3)
    iterations -- the number of times the operator is applied (default 1)
    """

    operations = ["erosion", "dilation", "opening", "closing"]
    if operation not in operations:
        raise ValueError("Invalid operation type. Expected one of: %s" % operations)
    if (kernel_size % 2) == 0:
        raise ValueError("The kernel must be of odd size (e.g., 3, 5, 7)")

    # Create a square matrix of size `kernel_size` as the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def erode(image, kernel = kernel, iterations = iterations):
        for _ in range(iterations):
            image = gf(image, np.min, footprint = kernel)
        return image

    def dilate(image, kernel = kernel, iterations = iterations):
        for _ in range(iterations):
            image = gf(image, np.max, footprint=kernel)
        return image

    print("Performing %s" % operation)
    if operation == "erosion":
        return erode(image)
    elif operation == "dilation":
        return dilate(image)
    elif operation == "opening":
        # opening: the dilation of the erosion
        arr = dilate(erode(image))
        arr[image != 1] = 0
        return arr
    else:
        # closing: the erosion of the dilation
        arr = erode(dilate(image))
        arr[image != 1] = 0
        return arr

"""
    Assumes a torch.tensor that is scaled from -1 to 1
"""
def tensor_to_rgb(a, rgb, clip_bounds=[0,0.5], gamma=1):
    rgb_array = np.transpose(a[rgb, :, :].numpy().astype(float), (1,2,0)) + 1 # 
    rgb_array = ((np.clip(rgb_array, *clip_bounds) - clip_bounds[0]) /
                                (clip_bounds[1] - clip_bounds[0])) ** gamma * 255 
    img = Image.fromarray(np.round(rgb_array).astype(np.uint8))

    return img


def plot_image(img, figsize=(8,8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, interpolation='nearest')
    