import requests
from PIL import Image
from azure.storage.blob import BlobClient
import folium
import numpy as np
import datetime
import pandas as pd
import sys

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
