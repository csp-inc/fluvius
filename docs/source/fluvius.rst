Classes
=======

.. autoclass:: src.fluvius.USGS_Water_DB
   :members: __init__

.. autoclass:: src.fluvius.USGS_Station
   :members: __init__

.. autoclass:: src.fluvius.WaterData
   :members: __init__, get_available_station_list, get_source_df, 
      get_station_data 
      
.. autoclass:: src.fluvius.WaterStation
   :members: __init__, get_area_of_interest, drop_bad_usgs_obs, build_catalog,
      get_cloud_filtered_image_df, merge_image_df_with_samples, get_scl_chip,
      get_visual_chip, get_io_lulc_chip, get_spectral_chip, get_chip_metadata,
      perform_chip_cloud_analysis, chip_cloud_analysis, get_chip_features

.. autoclass:: src.utils.MultipleRegression
   :members: __init__
