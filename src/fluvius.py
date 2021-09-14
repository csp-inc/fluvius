#web scrapping libraries
from bs4 import BeautifulSoup as bs
from pandas.core.dtypes.missing import na_value_for_dtype
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
#data processing libraries
from shapely.geometry import Point
import fsspec
import os
import folium
import time
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio import features
from rasterio import warp
from rasterio import windows
from rasterio.enums import Resampling
import concurrent.futures
from PIL import Image
import matplotlib.pyplot as plt
#planetary computer libraries
from pystac_client import Client
import planetary_computer as pc
from pystac.extensions.eo import EOExtension as eo
from azure.storage.blob import BlobClient
import traceback

BANDS_10M = ['AOT', 'B02', 'B03', 'B04', 'B08', 'WVP']
BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11']

EMPTY_METADATA_DICT = {
    "mean_viewing_azimuth": np.nan,
    "mean_viewing_zenith": np.nan,
    "mean_solar_azimuth": np.nan,
    "mean_solar_zenith": np.nan,
    "sensing_time": pd.NaT
}

BAD_USGS_COLS = ["Instantaneous computed discharge (cfs)_x", 
                 "Instantaneous computed discharge (cfs)_y"]

class USGS_Water_DB:
    def __init__(self, verbose=False):
        self.source_url = 'https://nrtwq.usgs.gov'
        self.verbose = verbose
        self.create_driver()

    def create_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        self.driver = driver

    def get_station_df(self):
        soup = self.get_url_text(self.source_url)
        js = str(soup.findAll('script')[6])
        marker_text_raw = js.split('L.marker')[1:-1]
        self.station_df = pd.concat([self.get_marker_info(m) for m in marker_text_raw]).reset_index(drop=True)

    def get_url_text(self, url):
        self.driver.get(url)
        result = requests.get(url, allow_redirects=False)
        if result.status_code==200:
            if self.verbose:
                print(f'Data found at {url}!')
            soup = bs(result.text, 'html.parser') 
            return soup
        else:
            if self.verbose:
                print(f'{url} response not 202!')
        return None

    def process_soup(self, soup):
        data_raw = str(soup).split('\n')
        data_raw = [elem for elem in data_raw if not ('#' in elem)]
        data_split = [d.split('\t') for d in data_raw]
        y = (i for i,v in enumerate(data_split) if ('' in v))
        stop = next(y)
        cols = data_split[0]
        units = data_split[1]
        columns = [f'{c} ({u})' if ' ' not in u else f'{c}' for c,u in zip(cols,units) ]
        data = data_split[2:stop]
        df = pd.DataFrame(data=data, columns=columns)
        return df

    def get_marker_info(self, marker_text):
        site_no = marker_text.split('site_no=')[1].split('>')[0].replace('"','')
        point = [float(p) for p in marker_text.split('[')[1].split(']')[0].split(',')]
        lat = point[0]
        lon = point[1]
        site_name = marker_text.split('<hr>')[1].split('<br')[0]
        df = pd.DataFrame([{'site_no':site_no,'site_name':site_name,'Latitude':lat,'Longitude':lon}])
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude,df.Latitude))

class USGS_Station:

    def __init__(self, site_no, instantaneous=False, verbose=False, year_range=np.arange(2013,2022)):
        self.site_no = site_no
        self.instantaneous = instantaneous
        self.verbose = verbose
        self.year_range = year_range
        self.create_driver()

    def create_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    def get_water_url(self, attribute, year):
        pcode_list = {'discharge':'00060',\
        'turbidity':'63680',\
        'temperature':'00010',\
        'dissolved_oxygen':'00300',\
        'ssd':'99409'}
        url_header = 'https://nrtwq.usgs.gov/explore/datatable?'
        timestep = 'uv'
        period = f'{year}_all'
        l = {'url_header':url_header, 'site_no':self.site_no, 'timestep':timestep}
        l['period'] = period
        l['pcode'] = pcode_list[attribute]
        url = f"{l['url_header']}site_no={l['site_no']}&pcode={l['pcode']}&period={l['period']}&timestep={l['timestep']}&format=rdb&is_verbose=y"
        return url

    def get_url_text(self, url):
        self.driver.get(url)
        result = requests.get(url, allow_redirects=False)
        if result.status_code==200:
            if self.verbose:
                print('Data found!')
            soup = bs(result.text, 'html.parser') 
            return soup
        else:
            if self.verbose:
                print('Data does not exist')
            return None

    def process_soup(self,soup,attribute):
        #might need to update this method to include instantaneous measurements
        if ((self.instantaneous) & (attribute=='ssd')):
            data_raw = str(soup).split('Discrete (laboratory-analyzed)')[1].split('\n')
            data_raw = [elem for elem in data_raw if not (' data' in elem)]
        else:
            data_raw = str(soup).split('\n')
        data_raw = [elem for elem in data_raw if not ('#' in elem)]
        #could use regex here..
        data_split = [d.split('\t') for d in data_raw]
        y = (i for i,v in enumerate(data_split) if ('' in v))
        stop = next(y)
        cols = data_split[0]
        units = data_split[1]
        columns = [f'{c} ({u})' if ' ' not in u else f'{c}' for c,u in zip(cols,units) ]
        data = data_split[2:stop]
        df = pd.DataFrame(data=data, columns=columns)
        return df

    def get_water_attribute(self, attribute, year):
        water_url = self.get_water_url(attribute, year)
        textsoup = self.get_url_text(water_url)
        out = None
        if textsoup is not None:
            out = self.process_soup(textsoup, attribute) 
        return out 

    def get_water_df(self, sleep_time=3, write_to_csv=False):
        #check if the csv file exists, if not download it...
        d = []
        for year in self.year_range:
            try:
                time.sleep(sleep_time)
                ssd_df = self.get_water_attribute('ssd', year)
                time.sleep(sleep_time)
                d_df = self.get_water_attribute('discharge', year)
                merged = d_df.merge(ssd_df, on='Date-Time')
                if d_df is not None:
                    if self.verbose:
                        print(f'Found {year} {self.site_no} data!') 
                    d.append(merged)
            except: #need to work on this a little bit to check for data or at least build method
                if self.verbose:
                    print(f'Timed out for {self.site_no}, year {year}!')
                continue #could time out due to no data available
        if d:
            self.df = pd.concat(d, ignore_index=True).dropna()
            if write_to_csv:
                sitefile = f'/content/data/{self.site_no}_data.csv' 
                self.df.to_csv(sitefile, index=False)
                print(f'Wrote {sitefile}!')
        else:
            self.df = None

class WaterData:

    def __init__(self, data_source, container, storage_options):
        self.container = container
        self.data_source = data_source
        self.storage_options = {'account_name':storage_options['account_name'],\
                                'account_key':storage_options['account_key']}
        self.filesystem = 'az'
        self.station_path = f'{self.container}/stations'
        self.station = {}

    def get_available_station_list(self):
        '''
        Searches the blob container and saves the list of directories
        '''
        fs = fsspec.filesystem(self.filesystem, \
                account_name=self.storage_options['account_name'], \
                account_key=self.storage_options['account_key'])
        return fs.ls(f'{self.station_path}')
    
    def get_source_df(self):
        '''
        Returns the station dataframe <pandas.DataFrame>
        '''
        source_url = f'az://{self.container}/{self.data_source}_station_metadata.csv'
        source_df = pd.read_csv(source_url, storage_options=self.storage_options) 
        #fs_list = self.get_available_station_list()
        #station_list = pd.DataFrame({'site_no':filter(None,\
        #                                   map(lambda sub:(''.join([ele for ele in sub])), fs_list))})

        crs = 'epsg:4326'
        if self.data_source=='usgs':
            source_df['site_no'] = [str(f).zfill(8) for f in source_df['site_no']]
        else:
            source_df['site_no'] = source_df['site_no'].astype(str)

        gdf = gpd.GeoDataFrame(source_df,\
                               geometry=gpd.points_from_xy(source_df.Longitude, source_df.Latitude),\
                               crs=crs)
        self.df = gdf

    def apply_buffer_to_points(self, buffer_distance, buffer_type='square', resolution=1):
        buffer_style = {'round':1, 'flat':2, 'square':3}
        srcdf = self.df.copy()
        srcdf = srcdf.to_crs('EPSG:3857')
        srcdf = srcdf.geometry.buffer(buffer_distance,\
                                      cap_style=buffer_style[buffer_type],\
                                      resolution=resolution)
        srcdf = srcdf.to_crs('EPSG:4326')
        self.df['buffer_geometry'] = srcdf.geometry

    def generate_map(self):
        '''
        plots web map using folium
        '''
        x, y = [],[]
        for p in self.df.geometry:
                x.append(p.x)
                y.append(p.y)
                cx = sum(x) / len(x)
                cy = sum(y) / len(y)
        self.plot_map = folium.Map(location=[cy, cx],\
            zoom_start=4,\
            tiles='CartoDB positron')
        for _, r in self.df.iterrows():
            folium.Marker(location=[r['Latitude'], r['Longitude']],\
                #popup=f"{r['site_no']}:\n{r['station_name']}").add_to(self.plot_map)
                popup=f"Site: {r['site_no']}").add_to(self.plot_map)
            polygons = gpd.GeoSeries(r['buffer_geometry'])
            geo_j = polygons.to_json()
            geo_j = folium.GeoJson(data=geo_j,style_function=lambda x: {'fillColor': 'orange'})
            geo_j.add_to(self.plot_map)
        tile = folium.TileLayer(\
                tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\
                attr = 'Esri',\
                name = 'Esri Satellite',\
                overlay = False,\
                control = True).add_to(self.plot_map)

    def get_space_bounds(self, geometry):
        coordinates =  np.dstack(geometry.boundary.coords.xy).tolist()
        area_of_interest = {
        "type": "Polygon",
        "coordinates": coordinates,
        }
        return area_of_interest

    def get_station_data(self, station=None):
        '''
        gets all the station data if station is None.
        '''
        if any(self.df['site_no'] == str(station)):
            geometry =  self.df[self.df['site_no'] == str(station)].buffer_geometry.iloc[0]
            aoi = self.get_space_bounds(geometry)
            ws  = WaterStation(str(station),\
                    aoi,\
                    self.container,\
                    self.storage_options,\
                    self.data_source)
            self.station[str(station)] = ws
        elif station is None:
            for s in self.df['site_no']:
                #use recursion to get all station data
                self.get_station_data(str(s))
        else:
            print('Invalid station name!')
        self.sort_station_data()
        
    def sort_station_data(self):
        self.station = {key: value for key, value in sorted(self.station.items())}


class WaterStation:
    ''' 
    Generalized water station data. May make child class for USGS, ANA, and ITV
    '''
    def __init__(self, site_no, area_of_interest, container, storage_options, data_source):
        self.site_no = site_no
        self.area_of_interest = area_of_interest
        self.container = container
        self.storage_options = storage_options
        self.data_source = data_source
        self.src_url = f'az://{container}/stations/{str(site_no).zfill(8)}.csv'
        self.df = pd.read_csv(self.src_url, storage_options=self.storage_options).dropna()
        self.get_time_bounds()
        sample_ids = [f'{str(site_no).zfill(8)}_{s:08d}' for s in (1+np.arange(len(self.df)))]
        #drop duplicates
        self.df.insert(0,'sample_id',sample_ids)
        self.df = self.df.drop_duplicates(subset='Date-Time')

    def format_time(self):
        self.df['Date-Time'] = pd.to_datetime(self.df['Date-Time'])
        self.df['Date-Time'] = self.df['Date-Time'].dt.date
        self.df = self.df.sort_values(by='Date-Time')

    def get_time_bounds(self):
        self.format_time()
        start = self.df['Date-Time'].iloc[0].strftime('%Y-%m-%d')
        end = self.df['Date-Time'].iloc[-1].strftime('%Y-%m-%d')
        self.time_of_interest = f'{start}/{end}'

    def drop_bad_usgs_obs(self):
        """
        Some stations from USGS have two measurements of instantaneous computed
        discharge. This method drops observations for which the two measurements
        are not equal. Note that the method only applies to "usgs" stations. If 
        WaterStation.data_source is not equal to "usgs", the method does nothing,
        so it can be safely applied to WaterStations from any data source with
        minimal performance impact.
        """
        if self.data_source == "usgs":
            df_cols = set(list(self.df.columns))
            if set(BAD_USGS_COLS).issubset(df_cols):
                bad_rows = self.df[BAD_USGS_COLS[0]] != self.df[BAD_USGS_COLS[1]]
                bad_idx = bad_rows[bad_rows].index
                self.df.drop(index=bad_idx, inplace=True)
                self.df.drop(BAD_USGS_COLS[1], axis=1, inplace=True)
                self.df.rename(
                    columns={BAD_USGS_COLS[0]: "Instantaneous computed discharge (cfs)"},
                    inplace=True
                )

    def build_catalog(self, collection='sentinel-2-l2a'):
        ''' 
        Use pystac-client to search for Sentinel 2 L2A data
        '''
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        print(f'building catalog for station {self.site_no} with {collection}!')

        search = catalog.search(
            collections=[collection], 
            intersects=self.area_of_interest, 
            datetime=self.time_of_interest    
            )
        matched_list = list(search.get_items())
        print(f"{len(matched_list)} Items found")
        if  len(matched_list) == 0:
            self.catalog = None
        else:
            self.catalog = search

    def get_cloud_filtered_image_df(self, cloud_thr):
        if not hasattr(self, 'catalog'):
            self.build_catalog()
        scene_list = sorted(self.catalog.get_items(), key=lambda item: eo.ext(item).cloud_cover)
        cloud_list = pd.DataFrame([{'Date-Time':s.datetime.strftime('%Y-%m-%d'),\
                'Tile Cloud Cover':eo.ext(s).cloud_cover,\
                'AOT-href':s.assets['AOT'].href,
                'B02-href':s.assets['B02'].href,
                'B03-href':s.assets['B03'].href,
                'B04-href':s.assets['B04'].href,
                'B05-href':s.assets['B05'].href,
                'B06-href':s.assets['B06'].href,
                'B07-href':s.assets['B07'].href,
                'B08-href':s.assets['B08'].href,
                'B11-href':s.assets['B11'].href,
                'B07-href':s.assets['B12'].href,
                'B8A-href':s.assets['B8A'].href,
                'WVP-href':s.assets['WVP'].href,
                'visual-href':s.assets['visual'].href,
                'scl-href':s.assets['SCL'].href,
                'meta-href': s.assets['granule-metadata'].href} \
                for s in scene_list if eo.ext(s).cloud_cover<cloud_thr])
        cloud_list['Date-Time'] = pd.to_datetime(cloud_list['Date-Time'])
        cloud_list['Date-Time'] = cloud_list['Date-Time'].dt.date
        self.image_df = cloud_list.sort_values(by='Date-Time')

    def merge_image_df_with_samples(self, day_tolerance=8):
        self.merged_df = self.df.copy()
        self.merged_df['Date'] = pd.to_datetime(self.merged_df['Date-Time'])
        self.image_df['Date'] = pd.to_datetime(self.image_df['Date-Time'])
        self.merged_df = pd.merge_asof(self.merged_df.sort_values(by='Date'),\
            self.image_df.sort_values(by='Date'),\
            on='Date',\
            suffixes=('', '_Remote'),\
            tolerance=pd.Timedelta(day_tolerance, 'days')).dropna()
        self.merged_df['InSitu_Satellite_Diff'] = \
                self.merged_df['Date-Time']-self.merged_df['Date-Time_Remote']
        self.total_matched_images = len(self.merged_df)
                   
    def get_scl_chip(self, signed_url, return_meta_transform=False):
        with rio.open(signed_url) as ds:    
            aoi_bounds = features.bounds(self.area_of_interest)
            warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
            self.scl_window = windows.from_bounds(
                transform=ds.transform,
                *warped_aoi_bounds
            ).round_lengths()
            band_data = ds.read(window=self.scl_window)
            scl = band_data[0]
            if return_meta_transform:
                out_meta = ds.meta
                out_transform = ds.window_transform(self.scl_window)
                return scl, out_meta, out_transform
            return scl

    def get_visual_chip(self, signed_url, return_meta_transform=False):
        with rio.open(signed_url) as ds:    
            aoi_bounds = features.bounds(self.area_of_interest)
            warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data = ds.read(window=aoi_window)
            self.chip_shape_10m = tuple([1] + list(band_data.shape[1:3])) # need 1st dim to be one since other chips are one band
            img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
            if return_meta_transform:
                out_meta = ds.meta
                out_transform = ds.window_transform(aoi_window)
                return img, out_meta, out_transform
            return img

    def get_spectral_chip(self, hrefs_10m, hrefs_20m, return_meta_transform=False):
        """
        Returns an image with one or more bands along the 3rd dimension from a signed url (one for each band)
        Args:
            hrefs_10m (list): hrefs for the 10 meter Sentinel bands
            hrefs_20m (list): hrefs for the 20 meter Sentinel bands
        """
        band_data_10m = list()
        for href in hrefs_10m:
            signed_href = pc.sign(href)
            with rio.open(signed_href) as ds:
                aoi_window = windows.Window(
                    self.scl_window.col_off * 2,
                    self.scl_window.row_off * 2,
                    self.scl_window.width * 2,
                    self.scl_window.height * 2
                )
                band_data_10m.append(ds.read(window=aoi_window))
                if href == hrefs_10m[-1] and return_meta_transform:
                    out_meta = ds.meta
                    out_transform = ds.window_transform(aoi_window)
        
        self.chip_shape_10m = band_data_10m[0].shape

        band_data_20m = list()
        for href in hrefs_20m:
            signed_href = pc.sign(href)
            with rio.open(signed_href) as ds:
                band_data_20m.append(ds.read(out_shape=self.chip_shape_10m,
                                             window=self.scl_window,
                                             resampling=Resampling.nearest))

        bands_array = np.transpose(np.concatenate(band_data_10m + band_data_20m, axis=0), axes=[1, 2, 0])

        if return_meta_transform:
            return bands_array, out_meta, out_transform
        else:
            return bands_array

    def get_chip_metadata(self, signed_url):
        req = requests.get(signed_url)
        soup = bs(req.text, "xml")
        mean_viewing_angles = soup.find_all("Mean_Viewing_Incidence_Angle")
        mean_viewing_azimuth = np.mean([float(angles.find("AZIMUTH_ANGLE").get_text()) 
                                        for angles in mean_viewing_angles])
        mean_viewing_zenith = np.mean([float(angles.find("ZENITH_ANGLE").get_text()) 
                                    for angles in mean_viewing_angles])

        mean_sun_angles = soup.find("Mean_Sun_Angle")
        mean_solar_zenith = float(mean_sun_angles.find("ZENITH_ANGLE").get_text())
        mean_solar_azimuth = float(mean_sun_angles.find("AZIMUTH_ANGLE").get_text())

        sensing_time = pd.to_datetime(soup.find("SENSING_TIME").get_text())

        meta_attributes = {
            "mean_viewing_azimuth": mean_viewing_azimuth,
            "mean_viewing_zenith": mean_viewing_zenith,
            "mean_solar_azimuth": mean_solar_azimuth,
            "mean_solar_zenith": mean_solar_zenith,
            "sensing_time": sensing_time
        }

        return meta_attributes

    def perform_chip_cloud_analysis(self, quiet=False):
        #probably can perform this in parallel
        '''
        #with concurrent.futures.ProcessPoolExecutor() as executor:
        #dt_list = [chip_cloud_analyisis(dt) for dt in executor.map(get_scl_chip )]
        '''
        chip_clouds_list = []
        for i, sc in self.merged_df.iterrows():
            try:
                print(f"Performing chip cloud analysis for sample {sc['sample_id']}")
                chip_clouds = self.chip_cloud_analysis(self.get_scl_chip(pc.sign(sc['scl-href'])))
                chip_clouds_list.append(chip_clouds)
            except:
                chip_clouds_list.append(np.nan)
                if not quiet:
                    print(f"{sc['scl-href']} cloud chip error!")
        self.merged_df['Chip Cloud Pct'] = chip_clouds_list
        
    def chip_cloud_analysis(self, scl):
        scl = np.array(scl)
        n_total_pxls = np.multiply(scl.shape[0], scl.shape[1])
        n_cloud_pxls = np.sum((scl>=7) & (scl<=10))
        chip_cloud_pct = 100*(n_cloud_pxls/n_total_pxls)
        return chip_cloud_pct

    def check_response(self,logfile='/content/log/response.log'):
        for i, scene_query in self.merged_df.iterrows():
            visual_href = pc.sign(scene_query['visual-href'])
            scl_href = pc.sign(scene_query['scl-href'])
            with open(logfile, 'a') as f:
                try:
                    vresponse = requests.get(visual_href)
                    print(f'{visual_href} returned {vresponse.status_code}', file=f)
                except:
                    print(f'{visual_href} returned {522}', file=f)
                try:
                    sresponse = requests.get(scl_href)
                    print(f'{scl_href} returned {sresponse.status_code}', file=f)
                except:
                    print(f'{visual_href} returned {522}', file=f)

    def get_chip_features(self, write_chips_to_blob=False, blob_root_dir=""):
        reflectances = []
        n_water_pixels = []
        metadata = []
        for i,scene_query in self.merged_df.iterrows():
            print(f"Extracting features for sample {scene_query['sample_id']}")
            hrefs_10m = scene_query[[band + "-href" for band in BANDS_10M]]
            hrefs_20m = scene_query[[band + "-href" for band in BANDS_20M]]
            scl_href = pc.sign(scene_query['scl-href'])
            meta_href = pc.sign(scene_query['meta-href'])
            n_bands = len(hrefs_10m + hrefs_20m)
            try:
                scl, scl_meta, scl_trans  = self.get_scl_chip(scl_href, True)
                scl = Image.fromarray(scl)
                # Resize scl (double its size to match img)
                scl = np.array(scl.resize(tuple([x*2 for x in scl.size]), Image.NEAREST))
                water_mask = ((scl==6) | (scl==2))
                # Excludes images with no water pixels
                if np.any(water_mask):
                    img, img_meta, img_trans = self.get_spectral_chip(hrefs_10m, hrefs_20m, True)
                    granule_metadata = self.get_chip_metadata(meta_href)
                    masked_array = img[water_mask, :]
                    mean_ref = np.nanmean(masked_array, axis=0)
                    reflectances.append(mean_ref)
                    n_water_pixels.append(np.sum(water_mask))
                    metadata.append(granule_metadata)

                    if write_chips_to_blob:
                        print()
                        self.write_chip_to_blob(
                            np.expand_dims(water_mask.astype(float),2),
                            scl_meta,
                            img_trans, # since it was resampled, proj is equal
                            "data/chips",
                            blob_root_dir,
                            self.site_no,
                            f"{scene_query['sample_id']}_{scene_query['Date-Time']}_water"
                        )
                        self.write_chip_to_blob(
                            img,
                            img_meta,
                            img_trans,
                            "data/chips",
                            blob_root_dir,
                            self.site_no,
                            f"{scene_query['sample_id']}_{scene_query['Date-Time']}"
                        )
                else: #no water pixels detected
                    reflectances.append([np.nan] * n_bands)
                    n_water_pixels.append(0)
                    metadata.append(EMPTY_METADATA_DICT)
            except:
                #print(f"{scene_query['visual-href']} returned response!")
                reflectances.append([np.nan] * n_bands)
                n_water_pixels.append(np.nan)
                metadata.append(EMPTY_METADATA_DICT)
            
        reflectances = np.array(reflectances)
        n_water_pixels = np.array(n_water_pixels)
        metadata = pd.DataFrame(metadata)
        
        collection = 'sentinel-2-l2a'
        bands = BANDS_10M + BANDS_20M
        for i,band in enumerate(bands):
            self.merged_df[f'{collection}_{band}'] = reflectances[:,i]

        self.merged_df['n_water_pixels'] = n_water_pixels
        self.merged_df = pd.concat([self.merged_df.reset_index(), metadata], axis = 1).set_index('index')

    def upload_local_to_blob(self, localfile, blobname):
        #blobclient = BlobClient.from_connection_string(conn_str=self.connection_string,\
        #                                            container_name=self.container,\
        #                                            blob_name=blobname)
        account_url = f"https://{self.storage_options['account_name']}.blob.core.windows.net"
        blobclient = BlobClient(account_url=account_url,\
                container_name="modeling-data",\
                blob_name=blobname,\
                credential=self.storage_options['account_key'])
        with open(f"{localfile}", "rb") as out_blob:
            blobclient.upload_blob(out_blob, overwrite=True) 

    def write_chip_to_blob(
            self,
            array,
            img_meta,
            img_transform,
            local_root_dir,
            blob_root_dir,
            site_no,
            sample_id, # sample_id = f"{scene_query['sample_id']}_{scene_query['Date-Time']}"
        ):
        img = np.moveaxis(array, -1, 0)
        h = img.shape[1]
        w = img.shape[2]
        c = img.shape[0]
        img_meta = {"driver": "GTiff",
                    "height": h,
                    "width": w,
                    "count": c,
                    "crs": img_meta["crs"],
                    "dtype": "uint16",
                    'transform': img_transform}
        if not os.path.exists(f'{local_root_dir}/{self.data_source}'):
            os.makedirs(f'{local_root_dir}/{self.data_source}')
        out_name = f'{local_root_dir}/{self.data_source}/{sample_id}.tif'
        blob_name = f'{blob_root_dir}/{self.data_source}/{sample_id}.tif'
        with rio.open(out_name, 'w', **img_meta) as dest:
            dest.write(img)
        self.upload_local_to_blob(out_name, blob_name)


    def visualize_chip(self, sample_id):
        scene_query = self.merged_df[self.merged_df['sample_id'] == sample_id]
        visual_href = pc.sign(scene_query['visual-href'].values[0])
        scl_href = pc.sign(scene_query['scl-href'].values[0])
        img = self.get_visual_chip(visual_href)
        scl = self.get_scl_chip(scl_href)
        water_mask = ((scl==6) | (scl==2))
        # masked_array = np.array(img)[water_mask, :]
        f, ax = plt.subplots(1,4, figsize=(20,20))
        cloud_mask = scl>7
        #extent = [self.bounds[0], self.bounds[2], self.bounds[1], self.bounds[3]]
        ax[0].imshow(img)
        ax[0].set_title('RGB Image')
        ax[0].axis('off')
        ax[1].imshow(scl,cmap='Accent')
        ax[1].set_title('Classification')
        ax[1].axis('off')
        ax[2].imshow(cloud_mask)
        ax[2].set_title('Cloud Mask')
        ax[2].axis('off')
        ax[3].imshow(water_mask,cmap='Blues')
        ax[3].set_title('Water Mask')
        ax[3].axis('off')
