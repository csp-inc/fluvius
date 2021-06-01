from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from shapely.geometry import Point
import fsspec
import pandas as pd
import geopandas as gpd
import folium
import re
import numpy as np
import time
import concurrent.futures

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
                print('Data found!')
            soup = bs(result.text, 'html.parser') 
            return soup
        else:
            if self.verbose:
                print('Data does not exist')
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

    def __init__(self, site_no, verbose=False, year_range=np.arange(2013,2022)):
        self.site_no = site_no
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

    def process_soup(self,soup):
        #might need to update this method to include instantaneous measurements
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
            out = self.process_soup(textsoup) 
        return out 
    def get_water_df(self, sleep_time=10, write_to_csv=False):
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
            self.df = pd.concat(d).dropna()
            if write_to_csv:
                sitefile = f'/content/data/{self.site_no}_data.csv' 
                self.df.to_csv(sitefile, index=False)
                print(f'Wrote {sitefile}!')
        else:
            self.df = None


class WaterStation:
    ''' 
    Generalized water station data. May make child class for USGS,ANA, and ITV
    '''
    def __init__(self, site_no, container, storage_options):
        self.site_no = site_no
        self.container = container
        self.storage_options = storage_options
        self.src_url = f'az://{container}/stations/{str(self.site_no)}.csv'
        self.df = pd.read_csv(src_url, storage_options=self.storage_options).dropna() 
        self.get_spacetime_bounds()

    def format_time(self):
        self.df['Date-Time'] = pd.to_datetime(self.df['Date-Time'])
        self.df['Date-Time'] = self.df['Date-Time'].dt.date
        self.df = self.df.sort_values(by='Date-Time')

    def get_spacetime_bounds(self):
        coordinates =  np.dstack(self.site_no.geometry.boundary.coords.xy).tolist()
        self.area_of_interest = {
        "type": "Polygon",
        "coordinates": coordinates,
        }
        start = df['Date-Time'].iloc[0].strftime('%Y-%m-%d')
        end = df['Date-Time'].iloc[-1].strftime('%Y-%m-%d')
        #use the date range of the data?
        #time_of_interest = "2019-06-01/2019-08-01"
        self.time_of_interest = f'{start}/{end}'

    def build_catalog(self, collection='sentinel-2-l2a'):
        ''' 
        Use pystac-client to search for Sentinel 2 L2A data
        '''
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        search = catalog.search(
            collections=[collection], 
            intersects=self.area_of_interest, 
            datetime=self.time_of_interest    
            )
        print(f"{search.matched()} Items found")
        self.catalog = search

    def get_scl_chip(self, signed_url, write_to_filename=None):
        with rasterio.open(signed_url) as ds:    
            aoi_bounds = features.bounds(self.area_of_interest)
            warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data = ds.read(window=aoi_window)
            scl = band_data[0].repeat(2, axis=0).repeat(2, axis=1)
            return scl

    def get_visual_chip(self, signed_url, write_to_filename=None):
        with rasterio.open(signed_url) as ds:    
            aoi_bounds = features.bounds(self.area_of_interest)
            warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data = ds.read(window=aoi_window)
            img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
            #w = img.size[0]; h = img.size[1]; aspect = w/h
            #target_w = scl.shape[1]; target_h = scl.shape[0]
            #img = img.resize((target_w,target_h),Image.BILINEAR)
            return img

    def get_filtered_cloud_list(search, cloud_thr):
        scene_list = sorted(search.items(), key=lambda item: item.ext.eo.cloud_cover)
        cloud_list = pd.DataFrame([{'Date-Time':s.datetime.strftime('%Y-%m-%d'),\
                'Cloud Cover':s.ext.eo.cloud_cover,\
                'visual-href':s.assets['visual-10m'].href,\
                'scl-href':s.assets['SCL-20m'].href} for s in scene_list if s.ext.eo.cloud_cover<cloud_thr])
        cloud_list['Date-Time'] = pd.to_datetime(cloud_list['Date-Time'])
        return cloud_list

    def chip_cloud_analysis(scl):
        n_total_pxls = np.multiply(scl.shape[0], scl.shape[1])
        n_cloud_pxls = np.sum((scl>=7) & (scl<=10))
        chip_cloud_pct = 100*(n_cloud_pxls/n_total_pxls)
        return chip_cloud_pct

    def get_reflectances(df):
        reflectances=[]
        for i,scene_query in df.iterrows():
            visual_href = pc.sign(scene_query['visual-href'])
            scl_href = pc.sign(scene_query['scl-href'])
            with rasterio.open(scl_href) as ds:    
                aoi_bounds = features.bounds(area_of_interest)
                warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
                aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
                band_data = ds.read(window=aoi_window)
                scl = band_data[0].repeat(2, axis=0).repeat(2, axis=1)

            with rasterio.open(visual_href) as ds:    
                aoi_bounds = features.bounds(area_of_interest)
                warped_aoi_bounds = warp.transform_bounds('epsg:4326', ds.crs, *aoi_bounds)
                aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
                band_data = ds.read(window=aoi_window)
                img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
                w = img.size[0]; h = img.size[1]; aspect = w/h
                target_w = scl.shape[1]; target_h = scl.shape[0]
                img = img.resize((target_w,target_h),Image.BILINEAR)

            cloud_mask = scl>7
            water_mask = ((scl==6) | (scl==2))
            reflectances.append(np.mean(np.array(img)[water_mask],axis=0))
        return np.array(reflectances)

class WaterData:

    def __init__(self, data_source, container, storage_options):
        self.container = container
        self.data_source = data_source
        self.storage_options = storage_options
        self.filesystem = 'az'
        self.station_path = f'{self.container}-data/stations'

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

        crs = {'init': 'epsg:4326'}

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
            zoom_start=7,\
            tiles='CartoDB positron')
        for _, r in self.df.iterrows():
            folium.Marker(location=[r['Latitude'], r['Longitude']],\
                #popup=f"{r['site_no']}:\n{r['station_name']}").add_to(self.plot_map)
                popup=f"Site:{r['site_no']}").add_to(self.plot_map)
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

