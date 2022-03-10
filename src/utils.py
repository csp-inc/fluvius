import requests
from PIL import Image
from azure.storage.blob import BlobClient
import folium
import numpy as np
import datetime
import pandas as pd
import json
import fsspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.ndimage.filters import generic_filter as gf

class MultipleRegression(nn.Module):
    """A custom class for use in PyTorch that builds an MLP model based on 
    user-supplied parameters.
    """

    def __init__(self, num_features, layer_out_neurons, activation_function):
        """Initializes the MLP based on user-provided parameters.

        Parameters
        ----------
        num_features : int
            The number of features for the model (i.e. the number of neurons in 
            the first layer)
        layer_out_neurons : list of int
            A list of length equal to the desired number of hidden layers in the 
            MLP, with elements corresponding to the number of neurons desired for
            each layer.
        activation_function : function
            The function (from torch.nn) to use for activation layers in the MLP. 
        """

        super(MultipleRegression, self).__init__()

        self.layer_out_neurons = layer_out_neurons
        n_layers = len(layer_out_neurons)
        self.n_layers = n_layers

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
        self.activate = activation_function


    def forward(self, inputs):
        x = self.activate(self.layer_1(inputs))
        for i in range(2, self.n_layers + 1):
            x = self.activate(getattr(self, f"layer_{i}")(x))
        x = self.layer_out(x)

        return (x)


def normalized_diff(b1, b2):
    b1 = b1.astype(float)
    b2 = b2.astype(float)
    return (b1 - b2) / (b1 + b2)


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
        activation_function=nn.PReLU(num_parameters=1),
        buffer_distance=500,
        day_tolerance=8,
        cloud_thr=80,
        mask_method1="lulc",
        mask_method2="mndwi",
        min_water_pixels=20,
        layer_out_neurons=[4, 4, 2],
        weight_decay=1e-2,
        n_folds=5,
        seed=123,
        verbose=True
    ):   
    """Fit an MLP model using crossfold validation. This function is used to run
    models for the hyperparameter grid search.

    Parameters
    ----------
    features : list of str
        A list of strings corresponding to the features that 
        should be used for model training. Must contain a subset of the following:
        ["sentinel-2-l2a_AOT", "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", 
        "sentinel-2-l2a_B04", "sentinel-2-l2a_B08", "sentinel-2-l2a_WVP", 
        "sentinel-2-l2a_B05", "sentinel-2-l2a_B06", "sentinel-2-l2a_B07", 
        "sentinel-2-l2a_B8A", "sentinel-2-l2a_B11", "sentinel-2-l2a_B12", 
        "mean_viewing_azimuth", "mean_viewing_zenith", "mean_solar_azimuth",
        "is_brazil"]
    learning_rate : float
        The starting learning rate to use for training.
    batch_size : int
        The batch size to use for training.
    epochs : int
        The number of training epochs to run.
    storage_options : dict
        A dictionary with the storage name and connection string to connect to 
        Azure blob storage
    activation_function : function, default=nn.PReLU(num_parameters=1)
        The function (from torch.nn) to use for activation layers in the MLP. 
    buffer_distance : int, default=500
        The buffer distance used for preprocessing training data (command line 
        arg to bin/ scripts).
    day_tolerance : int, default=8
        The maximum threshold used during data preprocessing for the number of 
        days between an observation and associated Sentinel 2 chip (command line
        arg to bin/ scripts).
    cloud_thr : float, default=80
        The percent of cloud cover (0-100) acceptable in the Sentinel tile corresponding
        to any given sample during data preprocessing. (command line arg to bin/
        scripts).
    mask_method1 : str, default="lulc"
        The primary mask method ("lulc" or "scl") used to prepare training data
        (command line arg to bin/ scripts).
    mask_method2 : str, default="mndwi"
        The secondary mask method ("mndwi", "ndvi", or "") used to prepare 
        training data (command line arg to bin/ scripts). 
    min_water_pixels : int, default=20
        The minimum number of water pixels used to calculate aggregate 
        reflectances for a given sample. Samples with fewer than this number of
        water pixels will not be used in training.
    layer_out_neurons : list of int, default=[4, 4, 2]
        A list of length equal to the desired number of hidden layers in the 
        MLP, with elements corresponding to the number of neurons desired for
        each layer.
    weight_decay : float, default=1e-2
        The weight decay to use when calculating loss.
    n_folds : int, default=5
        The number of folds in the training data (command line argument in bin/
        scripts).
    seed : int, default=123
        The seed used to initialize the pseudorandom number generator for use in
        partitioning data into train/validate folds and a separate test 
        partition.
    verbose : bool, default=True
        Should output on training progress be printed?
    """

    n_layers = len(layer_out_neurons)
    torch.set_num_threads(1)
    # Read the data
    fp = f"az://modeling-data/partitioned_feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mask_method1}{mask_method2}_masking_{n_folds}folds_seed{seed}.csv"
    data = pd.read_csv(fp, storage_options=storage_options)

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

        model = MultipleRegression(num_features, layer_out_neurons, activation_function)
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[7500, 10000, 12500, 14000],
            gamma=0.1
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

            # calculate R^2 for this epoch
            val_R2.append(r2_score(val_pred, y_val))

            scheduler.step()

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
        "batch_size": batch_size,
        "layer_out_neurons": layer_out_neurons,
        "epochs": epochs,
        "weight_decay": weight_decay,
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
        activation_function=nn.PReLU(num_parameters=1),
        buffer_distance=500,
        day_tolerance=8,
        cloud_thr=80,
        mask_method1="lulc",
        mask_method2="mndwi",
        min_water_pixels=20,
        layer_out_neurons=[24, 12, 6],
        weight_decay=1e-2,
        n_folds=5,
        seed=123,
        verbose=True
    ):
    """Fit an MLP model using the entire training set. This function is used to
    fit the top model identified from the grid search using all of the training
    data. Two files are written to Azure Blob storage: a model checkpoint 
    (.pt file), and a model metadata file (.json). Files are witten to the 
    model-outputs container in the fluviusdata storage account. NOTE: Running 
    this function will overwrite results on Azure Blob Storage, so use this 
    function with caution.

    Parameters
    ----------
    features : list of str
        A list of strings corresponding to the features that 
        should be used for model training. Must contain a subset of the following:
        `["sentinel-2-l2a_AOT", "sentinel-2-l2a_B02", "sentinel-2-l2a_B03", 
        "sentinel-2-l2a_B04", "sentinel-2-l2a_B08", "sentinel-2-l2a_WVP", 
        "sentinel-2-l2a_B05", "sentinel-2-l2a_B06", "sentinel-2-l2a_B07", 
        "sentinel-2-l2a_B8A", "sentinel-2-l2a_B11", "sentinel-2-l2a_B12", 
        "mean_viewing_azimuth", "mean_viewing_zenith", "mean_solar_azimuth",
        "is_brazil"]`
    learning_rate : float
        The starting learning rate to use for training.
    batch_size : int
        The batch size to use for training.
    epochs : int
        The number of training epochs to run.
    storage_options : dict
        A dictionary with the storage name and connection string to connect to 
        Azure blob storage
    activation_function : function, default=nn.PReLU(num_parameters=1)
        The function (from torch.nn) to use for activation layers in the MLP. 
    buffer_distance : int, default=500
        The buffer distance used for preprocessing training data (command line 
        arg to bin/ scripts).
    day_tolerance : int, default=8
        The maximum threshold used during data preprocessing for the number of 
        days between an observation and associated Sentinel 2 chip (command line
        arg to bin/ scripts).
    cloud_thr : float, default=80
        The percent of cloud cover (0-100) acceptable in the Sentinel tile corresponding
        to any given sample during data preprocessing. (command line arg to bin/
        scripts).
    mask_method1 : str, default="lulc"
        The primary mask method ("lulc" or "scl") used to prepare training data
        (command line arg to bin/ scripts).
    mask_method2 : str, default="mndwi"
        The secondary mask method ("mndwi", "ndvi", or "") used to prepare 
        training data (command line arg to bin/ scripts). 
    min_water_pixels : int, default=20
        The minimum number of water pixels used to calculate aggregate 
        reflectances for a given sample. Samples with fewer than this number of
        water pixels will not be used in training.
    layer_out_neurons : list of int, default=[4, 4, 2]
        A list of length equal to the desired number of hidden layers in the 
        MLP, with elements corresponding to the number of neurons desired for
        each layer.
    weight_decay : float, default=1e-2
        The weight decay to use when calculating loss.
    n_folds : int, default=5
        The number of folds in the training data (command line argument in bin/
        scripts).
    seed : int, default=123
        The seed used to initialize the pseudorandom number generator for use in
        partitioning data into train/validate folds and a separate test 
        partition.
    verbose : bool, default=True
        Should output on training progress be printed?
    """    
    n_layers = len(layer_out_neurons)

    # Read the data
    fp = f"az://modeling-data/partitioned_feature_data_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mask_method1}{mask_method2}_masking_{n_folds}folds_seed{seed}.csv"

    data = pd.read_csv(fp, storage_options=storage_options)
    data["Log SSC (mg/L)"] = np.log(data["SSC (mg/L)"])
    
    response = "Log SSC (mg/L)"
    not_enough_water = data["n_water_pixels"] < min_water_pixels
    data.drop(not_enough_water[not_enough_water].index, inplace=True)
    ssc_0 = data["SSC (mg/L)"] == 0
    data.drop(ssc_0[ssc_0].index, inplace=True)

    test = data[data["partition"] == "testing"]
    data = data[data["partition"] != "testing"]

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

    model = MultipleRegression(num_features, layer_out_neurons, activation_function)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[7500, 10000, 12500, 14000],
            gamma=0.1
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
        "batch_size": batch_size,
        "layer_out_neurons": layer_out_neurons,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "activation": f"{activation_function}",
        "train_loss": train_loss_list,
        "train_pooled_mse": train_loss_list[-1],
        "test_site_mse": test_site_mse,
        "test_pooled_mse": test_pooled_mse,
        "y_train_sample_id": list(data["sample_id"]),
        "y_test_sample_id": list(test["sample_id"]),
        #"X_test_scaled": X_test_scaled,
        "y_obs_train": list(y_train),
        "y_pred_train": train_pred_list,
        "y_obs_test": list(y_test),
        "y_pred_test": test_pred_list
    }

    # save the model!
    model_out_fn_prefix = f"top_model_buffer{buffer_distance}m_daytol{day_tolerance}_cloudthr{cloud_thr}percent_{mask_method1}{mask_method2}_masking_{n_folds}folds_seed{seed}"
    torch.save(model.state_dict(), f"/content/output/mlp/{model_out_fn_prefix}.pt")

    with open(f"/content/output/mlp/{model_out_fn_prefix}_metadata.json", 'w') as f:
        json.dump(output, f)

    fs = fsspec.filesystem("az", **storage_options)
    fs.put_file(f"/content/output/mlp/{model_out_fn_prefix}.pt", f"model-output/{model_out_fn_prefix}.pt", overwrite=True)
    fs.put_file(f"/content/output/mlp/{model_out_fn_prefix}_metadata.json", f"model-output/{model_out_fn_prefix}_metadata.json", overwrite=True)
    
    return output


def plot_obs_predict(obs_pred, title, units="log(SSC)", savefig=False, outfn=""):
    """
    Plot observations vs. Predictions:
    
    Arguments:
    obs_pred: A Pandas DataFrame containing a column of predictions called "pred"
        and a column of observations called "obs".
    title: String. The title for the plot
    units: String. The units/labal that should be used in axes.
    savefig: Boolean. Should the plot be saved
    outfn: String. If savefig is True, the file name where the plot should be
        saved
    """
    plt.figure(figsize=(8,8))
    max_val = np.maximum(np.max(obs_pred.iloc[:,0]), np.max(obs_pred.iloc[:,1]))
    plt.plot(list(range(0, round(max_val) + 1)), list(range(0,round(max_val) + 1)), color="black", label="One-to-one 1 line")
    plt.scatter(obs_pred["obs"], obs_pred["pred"])
    plt.xlabel(f"{units} Observed")
    plt.ylabel(f"{units} Predicted")
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


##### Pixel-level ssc prediction
import rasterio as rio, numpy as np, os, torch
from matplotlib import pyplot as plt
from PIL import Image

RIO_BANDS_ORDERED = {
    "sentinel-2-l2a_AOT":1, 
    "sentinel-2-l2a_B02":2, 
    "sentinel-2-l2a_B03":3, 
    "sentinel-2-l2a_B04":4, 
    "sentinel-2-l2a_B08":5, 
    "sentinel-2-l2a_WVP":6,
    "sentinel-2-l2a_B05":7, 
    "sentinel-2-l2a_B06":8, 
    "sentinel-2-l2a_B07":9, 
    "sentinel-2-l2a_B8A":10, 
    "sentinel-2-l2a_B11":11, 
    "sentinel-2-l2a_B12":12
}

RGB_MIN = 0
RGB_MAX = 6000
GAMMA = 0.6

def predict_pixel_ssc(sentinel_values, sentinel_features, non_sentinel_values, non_sentinel_features, all_features, scaler, model):
    obs_dict = dict.fromkeys(all_features)
    obs_dict.update(zip(sentinel_features, sentinel_values))
    obs_dict.update(zip(non_sentinel_features, non_sentinel_values))
    feature_values = torch.Tensor(
        list(
            scaler.transform(
                np.array(
                    list(obs_dict.values()), ndmin=2
                    )
                )[0, :]
            )
        )
    with torch.no_grad():
        model.eval()
        y_pred = model(feature_values).squeeze().numpy()

    return np.exp(y_pred.item())


def overlay_ssc_img(img, water, ssc_pixel_predictions, has_aot, cramp="hot"):

    if has_aot:
        rgb = [3,2,1]
    else:
        rgb = [2,1,0]

    cm = plt.get_cmap(cramp)
    ssc = (cm(np.interp(ssc_pixel_predictions, (5, 100), (0, 1))) * 255).astype(np.uint8)[:, :, 0:3]
    img2 =  np.moveaxis(
        np.interp(
            np.clip(
                img[rgb, :, :], # Models always included RGB bands, which are always positioned a 3,2,1 in the list of features for each model -- beware, this is hard coded
                RGB_MIN,
                RGB_MAX
            ), 
            (RGB_MIN, RGB_MAX),
            (0, 1)
        ) ** GAMMA * 255,
        0,
        2
    ).astype(np.uint8)

    img2[(water == True), :] = ssc[(water == True), :]

    return img2

# # The following will need to be run at the top-level scope prior to using this
# # function in order for it to work
# with open("/content/credentials") as f:
#     env_vars = f.read().split("\n")

# for var in env_vars:
#     key, value = var.split(" = ")
#     os.environ[key] = value
def predict_chip(features, sentinel_features, non_sentinel_features, observation, model, scaler):
    non_sentinel_values = list(observation[non_sentinel_features[0:]])

    with rio.Env(
        AZURE_STORAGE_ACCOUNT=os.environ["ACCOUNT_NAME"], 
        AZURE_STORAGE_ACCESS_KEY=os.environ["BLOB_KEY"]
    ):
        with rio.open(observation["raw_img_chip_href"]) as ds:
            img = ds.read([RIO_BANDS_ORDERED[x] for x in sentinel_features])
        with rio.open(observation["water_chip_href"]) as ds:
            water = ds.read(1)
    
    has_aot = "sentinel-2-l2a_AOT" in sentinel_features

    predictions = np.empty(water.shape)
    predictions[(water == False)] = np.NaN
    predictions[(water == True)] = np.apply_along_axis(
        predict_pixel_ssc, 0,
        img[:, (water == True)],
        sentinel_features,
        non_sentinel_values,
        non_sentinel_features,
        features,
        scaler,
        model)
    
    predictions = np.clip(predictions, a_min=0, a_max=np.nanquantile(predictions, 0.95))
    pred_chip = overlay_ssc_img(img, water, predictions, has_aot=has_aot, cramp="hot")

    return pred_chip 