from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
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

class USGS_Station_Data:
    def __init__(self):
        self.driver = create_driver()
    def get_station_data(self):
        pass


