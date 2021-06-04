#only considering spatial and temporal aspects not the connectivity
# %reset 
# import sys
# sys.modules[__name__].__dict__.clear()
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
%matplotlib inline
import numpy as np
from collections import Counter
from pyproj import Proj, transform
import itertools
from itertools import groupby
from operator import itemgetter
from sklearn.cluster import DBSCAN
import sklearn.cluster as cluster
import seaborn as sns
import time
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from geopy.distance import great_circle
from shapely.geometry import Point, MultiPoint
from matplotlib import cm
from scipy.spatial import ConvexHull
from functools import reduce
import itertools
import copy
import geopandas as gpd
import osmnx as ox
import networkx as nx
from tabpy.tabpy_tools.client import Client
 
#Data Processing and combining....................................................
#loading speed files for flooding, medium and normal days per 15 min data
flood = pd.read_csv('D:/BILAL/flood_project/flood_main_road.csv')

#CLustering.....................................................................................
#sample_data
# sample = flood[((flood['HH_ID']==7) | (flood['HH_ID']==8)) & (flood['ROUTE_DIR']==3502)]

#Dbscan calculation......................................................
def bilal_db(data,t1,t2,dire,diff,speed_x,speed_y):
    
    sample = data[((data['HH_ID']==t1) | (data['HH_ID']==t2)) & (data['ROUTE_DIR']==dire)]
    congested_links = sample[(sample['diff']<=-3) & (sample['TRVL_SPD_x']<=speed_x) & (sample['TRVL_SPD_y']>=speed_y)]
    congested_links['links_repeatition'] = congested_links['LINK_ID'].map(congested_links['LINK_ID'].value_counts()) #how many times each link is repeated in different time intervals
    congested_links = congested_links.drop_duplicates(subset=['LINK_ID'])
    min_samples = 4
    best_val = 0.65
    kms_per_radian = 6371.0088
    # rec_dist = best_val*kms_per_radian
    rec_dist = best_val
    #DBSCAN algorithm................
    dist = rec_dist # In KM
    epsilon = dist / kms_per_radian
    epsilon = dist / kms_per_radian
    #real attempt at dbscan
    model = DBSCAN(eps = epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    model.fit(np.radians(data[['O_y','O_x']])) #converting lat lon to radians as haversine takes values in radians
    return model.labels_.tolist()


# geopandas to visualize along with base map.............................
geo_df = gpd.GeoDataFrame(congested_links.drop(['diff'], axis=1), crs={'init': 'epsg:4326'}, geometry=[Point(xy) for xy in zip(congested_links.O_x, congested_links.O_y)])
# Set figure size
fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
# Import NYC Neighborhood Shape Files
daejeon_full = gpd.read_file('D:/daejeon-shapefile/daejeon-shapefile.shp')
daejeon_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', zorder=1)
geo_df.plot(ax=ax, column=geo_df['Clusters{}'.format(best_val)], alpha=0.5, cmap='viridis', linewidth=0.8, zorder=2)

z=[] #HULL simplices coordinates will be appended here
for i in range (0,num_clusters-1):
    dfq=geo_df[geo_df['Clusters{}'.format(best_val)]==i]
    Y = np.array(dfq[['O_y', 'O_x']])
    hull = ConvexHull(Y)
    plt.plot(Y[:, 1],Y[:, 0],  'o')
    z.append(Y[hull.vertices,:].tolist())
    for simplex in hull.simplices:
        ploted=plt.plot( Y[simplex, 1], Y[simplex, 0],'k-',c='m', zorder = 3)

#plotting without noises.......................................................................
cong_filter = congested_links[congested_links['Clusters{}'.format(best_val)]!=-1]
#geopandas to visualize along with base map
geo_df = gpd.GeoDataFrame(cong_filter.drop(['diff'], axis=1), crs={'init': 'epsg:4326'}, geometry=[Point(xy) for xy in zip(cong_filter.O_x, cong_filter.O_y)])
# Set figure size
fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
# Import NYC Neighborhood Shape Files
daejeon_full = gpd.read_file('D:/daejeon-shapefile/daejeon-shapefile.shp')
daejeon_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', zorder=1)
geo_df.plot(ax=ax, column=geo_df['Clusters{}'.format(best_val)], alpha=0.5, cmap='viridis', linewidth=0.8, zorder=2)

z=[] #HULL simplices coordinates will be appended here
for i in range (0,num_clusters-1):
    dfq=geo_df[geo_df['Clusters{}'.format(best_val)]==i]
    Y = np.array(dfq[['O_y', 'O_x']])
    hull = ConvexHull(Y)
    plt.plot(Y[:, 1],Y[:, 0],  'o')
    z.append(Y[hull.vertices,:].tolist())
    
    for simplex in hull.simplices:
        ploted=plt.plot( Y[simplex, 1], Y[simplex, 0],'k-',c='m', zorder = 3)

# congested_links['cl_links_repeatition'.format(best_val)] = congested_links.groupby(['Clusters{}'.format(best_val)])['links_repeatition'].transform('mean')

print("Epsilon value is",best_val,"KM,min num of points within limits are", min_samples,",number of clusters", num_clusters)


#cluster wise ranking..................................................................................

# output = congested_links
# output['cl_links_count'.format(best_val)] = output.groupby(['Clusters{}'.format(best_val)])['LINK_ID'].transform('count')
# output['cl_links_repeatition'.format(best_val)] = output.groupby(['Clusters{}'.format(best_val)])['links_repeatition'].transform('mean')
# output['Intensity'.format(best_val)] = output['cl_links_repeatition'] * output['cl_links_count']
# output = output[['O_x','O_y','Intensity','Clusters{}'.format(best_val)]]
# output.drop_duplicates('Clusters{}'.format(best_val), inplace = True)

#now find clusterwise mean
# output = output[['Clusters{}'.format(best_val), 'cl_links_repeatition'.format(best_val), 'cl_links_count'.format(best_val), 'Intensity'.format(best_val)]]
# output.drop_duplicates('Clusters{}'.format(best_val), inplace = True)
# output = round(output,2)
# print(output.to_latex(index=False))
# 

#save the output file
congested_links.to_csv('C:/Users/UserK/Desktop/flood_project/congested_links68_orig.csv', index = False)
# road_link_nodes_2.to_csv('C:/Users/UserK/Desktop/flood_project/raods.csv', index = False)
# flood_2.to_csv('C:/Users/UserK/Desktop/flood_project/flood.csv', index = False)




