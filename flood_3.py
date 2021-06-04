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
 
#Data Processing and combining....................................................
#loading speed files for flooding, medium and normal days per 15 min data
speed_1 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_15MN_20200730.csv')
speed_1 = speed_1.iloc[:,1:6]
speed_2 = pd.read_csv('D:/BILAL/5. Flooding/daejon network master data/MRT_BASE_TF_INFO_15MN_20200813.csv')
speed_2 = speed_2.iloc[:,1:6]
speed_12rse = pd.merge(speed_1,speed_2,on=['LINK_ID','HH_ID','MN_ID'], how='left')
#VDS speed data
speed_1v = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/VDS 5 min data/5.속도및교통량_5분(20200730).xlsx')
speed_1v.drop(index=0, inplace=True)
speed_1v = speed_1v.iloc[:,[1,2,3,7,8]]
speed_1v = speed_1v.rename({'VDS_TR_VOL_YMDH':'VDS_flood_V'},axis=1)
speed_2v = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/VDS 5 min data/6.속도및교통량_5분(20200813).xlsx')
speed_2v.drop(index=0, inplace=True)
speed_2v = speed_2v.iloc[:,[1,2,3,7,8]]
speed_2v = speed_2v.rename({'VDS_TR_VOL_YMDH':'VDS_normal_V'},axis=1)
#merging vds speeds and volume
speed_12v = pd.merge(speed_1v,speed_2v,on=['LINK_ID','HH_ID','MN_ID'], how='left')
#merging rse speeds and TT
speed = pd.merge(speed_12rse,speed_12v,on=['LINK_ID','HH_ID','MN_ID'], how='left')
#loading main master data which doesnt change
road_link = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/4.도로-링크매핑(d_road_route_link).xlsx')
road_link.drop(index=0, inplace=True)
road_link = road_link.iloc[:,0:4]
node = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/2.노드마스터(d_node).xlsx')
node.drop(index=0, inplace=True)
node['START_NODE_ID'] = node['NODE_ID']
node['END_NODE_ID'] = node['NODE_ID']
link = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/1.링크마스터(d_link).xlsx')
link.drop(index=0, inplace=True)
link = link.rename({'SEQ':'SEQ_link'}, axis = 1)
road = pd.read_excel('D:/BILAL/5. Flooding/daejon network master data/3.도로마스터(d_road_route).xlsx')
road.drop(index=0, inplace=True)
# road_link.sort_values(['ROAD_ROUTE_ID', 'SEQ'], ascending=[True, True], inplace=True)
road_link = pd.merge(road_link,road[['ROAD_ROUTE_ID','MAIN_ROAD_YN']], on = ['ROAD_ROUTE_ID'], how= 'left')
road_link = road_link.rename({'SEQ':'SEQ_route'},axis=1)
road_link_nodes = pd.merge(link[['LINK_ID', 'START_NODE_ID', 'END_NODE_ID', 'LINK_LEN','ROAD_NM','SEQ_link']],road_link,on='LINK_ID', how='left')
road_link_nodes_2 = pd.merge(road_link_nodes,node[['START_NODE_ID','POSX','POSY']],on='START_NODE_ID', how='left')
road_link_nodes_2 = pd.merge(road_link_nodes_2,node[['END_NODE_ID','POSX','POSY']],on='END_NODE_ID', how='left')
#Flooding day scenario
flood = pd.merge(speed,road_link_nodes_2,on='LINK_ID', how='left')
flood = flood[flood.POSX_x != 'T']
flood = flood[flood.POSX_y != 'T']
#transform origin coordinates to WGS84
inProj = Proj(init='epsg:5181') #5179 #5183 #5187 #5180 ##5186 #5185 #5184 #5181 is the real
outProj = Proj(init='epsg:4326')
x1,y1 = flood['POSX_x'],flood['POSY_x']
x2,y2 = transform(inProj,outProj,x1,y1)
flood['O_x'] = x2.tolist()
flood['O_y'] = y2.tolist()
#transform destination coordinates to WGS84
inProj = Proj(init='epsg:5181') #5179 #5183 #5187 #5180 ##5186 #5185 #5184 #5181 is the real
outProj = Proj(init='epsg:4326')
x1,y1 = flood['POSX_y'],flood['POSY_y']
x2,y2 = transform(inProj,outProj,x1,y1)
# print(x2,y2)
flood['D_x'] = x2.tolist()
flood['D_y'] = y2.tolist()
#original data
road_link_nodes_2 = road_link_nodes_2[road_link_nodes_2.POSX_x != 'T']
road_link_nodes_2 = road_link_nodes_2[road_link_nodes_2.POSX_y != 'T']
x1,y1 = road_link_nodes_2['POSX_x'],road_link_nodes_2['POSY_x']
x2,y2 = transform(inProj,outProj,x1,y1)
# print(x2,y2)
road_link_nodes_2['O_x'] = x2.tolist()
road_link_nodes_2['O_y'] = y2.tolist()
x1,y1 = road_link_nodes_2['POSX_y'],road_link_nodes_2['POSY_y']
x2,y2 = transform(inProj,outProj,x1,y1)
# print(x2,y2)
road_link_nodes_2['D_x'] = x2.tolist()
road_link_nodes_2['D_y'] = y2.tolist()
#removing unrequired columns
flood = flood.drop(['POSX_x', 'POSY_x', 'POSX_y', 'POSY_y'],1)
#data manipulation part
flood['TRVL_SPD_x'] = flood['TRVL_SPD_x']-4
#difference in speed of flood vs normal day
flood['diff'] = flood['TRVL_SPD_x']-flood['TRVL_SPD_y']
#plotting of speed data of flood and normal
plt.hist(flood['TRVL_SPD_y'])
plt.ylim(0, 800000)
plt.hist(flood['TRVL_SPD_x'])
plt.ylim(0, 800000)

#CLustering.....................................................................................
#sample_data
sample = flood[((flood['HH_ID']==7) | (flood['HH_ID']==8)) & (flood['ROUTE_DIR']==3502)]
sample_2 = sample.drop_duplicates(subset=['LINK_ID'])
sample_2['links_in_route'] = sample_2.groupby(['ROAD_ROUTE_ID'])['LINK_ID'].transform('count')
sample_2 = sample_2[['ROAD_ROUTE_ID','LINK_ID','links_in_route']]
#congested links
congested_links = sample[(sample['diff']<=-3) & (sample['TRVL_SPD_x']<=10) & (sample['TRVL_SPD_y']>=10)]
# congested_links = sample[(sample['diff']<=-10) & (sample['TRVL_SPD_x']!=30)]
congested_links['links_repeatition'] = congested_links['LINK_ID'].map(congested_links['LINK_ID'].value_counts()) #how many times each link is repeated in different time intervals
#remove the duplicated links on time interval
congested_links = congested_links.drop_duplicates(subset=['LINK_ID'])
congested_links['cong_links'] = congested_links.groupby(['ROAD_ROUTE_ID'])['LINK_ID'].transform('count')
#merge congested links and original sample
congested_links = congested_links.merge(sample_2.drop_duplicates(subset=['LINK_ID']), how='left')
congested_links = congested_links.sort_values('ROAD_ROUTE_ID')
congested_links['road_cong'] =  (congested_links['cong_links']/congested_links['links_in_route'])*100

#plotting of speed data of flood and normal
plt.hist(congested_links['TRVL_SPD_y'])
plt.ylim(0, 500)
plt.hist(congested_links['TRVL_SPD_x'])
plt.ylim(0, 500)

#nearest neighbours to find good epsilon value
# Calculate the average distance between each point in the data set and its 20 nearest neighbors (my selected MinPts value).
# “Knee method”. The goal is to find the average of distances for every point to its K nearest neighbors and select the distance at which maximum curvature or a sharp change happens. The value of K is set to be equal to minPoints.
min_samples = 6
#Elbow method
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(np.radians(congested_links[['O_y','O_x']]))
distances, indices = neighbors_fit.kneighbors(np.radians(congested_links[['O_y','O_x']]))
# Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances*6371.008)


#Dbscan calculation......................................................
best_val = 0.5
kms_per_radian = 6371.0088
# rec_dist = best_val*kms_per_radian
rec_dist = best_val
#DBSCAN algorithm................
dist = rec_dist # In KM
epsilon = dist / kms_per_radian
epsilon = dist / kms_per_radian
#real attempt at dbscan
model = DBSCAN(eps = epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
model.fit(np.radians(congested_links[['O_y','O_x']])) #converting lat lon to radians as haversine takes values in radians
cluster_labels = model.labels_
num_clusters = len(set(cluster_labels))
# clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
# clusters = clusters.drop(clusters.index[[len(clusters)-1]])
print('Number of clusters: {}'.format(num_clusters))
congested_links['Clusters'] = model.labels_
congested_links['Clusters'].value_counts()

#geopandas to visualize along with base map
geo_df = gpd.GeoDataFrame(congested_links.drop(['road_cong'], axis=1), crs={'init': 'epsg:4326'}, geometry=[Point(xy) for xy in zip(congested_links.O_x, congested_links.O_y)])
# Set figure size
fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
# Import NYC Neighborhood Shape Files
daejeon_full = gpd.read_file('D:/daejeon-shapefile/daejeon-shapefile.shp')
daejeon_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', zorder=1)
geo_df.plot(ax=ax, column=geo_df.Clusters, alpha=0.5, cmap='viridis', linewidth=0.8, zorder=2)


z=[] #HULL simplices coordinates will be appended here
for i in range (0,num_clusters-1):
    dfq=geo_df[geo_df['Clusters']==i]
    Y = np.array(dfq[['O_y', 'O_x']])
    hull = ConvexHull(Y)
    plt.plot(Y[:, 1],Y[:, 0],  'o')
    z.append(Y[hull.vertices,:].tolist())
    for simplex in hull.simplices:
        ploted=plt.plot( Y[simplex, 1], Y[simplex, 0],'k-',c='m', zorder = 3)

#plotting without noises.......................................................................
cong_filter = congested_links[congested_links['Clusters']!=-1]
#geopandas to visualize along with base map
geo_df = gpd.GeoDataFrame(cong_filter.drop(['road_cong'], axis=1), crs={'init': 'epsg:4326'}, geometry=[Point(xy) for xy in zip(cong_filter.O_x, cong_filter.O_y)])
# Set figure size
fig, ax = plt.subplots(figsize=(10,10))
ax.set_aspect('equal')
# Import NYC Neighborhood Shape Files
daejeon_full = gpd.read_file('D:/daejeon-shapefile/daejeon-shapefile.shp')
daejeon_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', zorder=1)
geo_df.plot(ax=ax, column=geo_df.Clusters, alpha=0.5, cmap='viridis', linewidth=0.8, zorder=2)

z=[] #HULL simplices coordinates will be appended here
for i in range (0,num_clusters-1):
    dfq=geo_df[geo_df['Clusters']==i]
    Y = np.array(dfq[['O_y', 'O_x']])
    hull = ConvexHull(Y)
    plt.plot(Y[:, 1],Y[:, 0],  'o')
    z.append(Y[hull.vertices,:].tolist())
    
    for simplex in hull.simplices:
        ploted=plt.plot( Y[simplex, 1], Y[simplex, 0],'k-',c='m', zorder = 3)
print("Epsilon value is",best_val,"KM,min num of points within limits are", min_samples,",number of clusters", num_clusters)

#cluster wise ranking..................................................................................
# congested_links['cl_links_repeatition'] = congested_links.groupby(['Clusters'])['links_repeatition'].transform('mean')
# # congested_links['cl_speed'] = congested_links.groupby(['Clusters'])['TRVL_SPD_x'].transform('mean')
# # congested_links['cl_road_cong'] = congested_links.groupby(['Clusters'])['road_cong'].transform('mean')
# congested_links['cl_links_count'] = congested_links.groupby(['Clusters'])['LINK_ID'].transform('count') # this count contains the repeated links too in different time intervals
# congested_links['Intensity'] = congested_links['cl_links_repeatition'] * congested_links['cl_links_count']

output = congested_links
output.drop_duplicates('LINK_ID', inplace = True)
output['cl_links_count'] = output.groupby(['Clusters'])['LINK_ID'].transform('count')
output['cl_links_repeatition'] = output.groupby(['Clusters'])['links_repeatition'].transform('mean')
output['Intensity'] = output['cl_links_repeatition'] * output['cl_links_count']

#now find clusterwise mean
output = output[['Clusters', 'cl_links_repeatition', 'cl_links_count', 'Intensity']]
output.drop_duplicates('Clusters', inplace = True)
output = round(output,2)
print(output.to_latex(index=False))

#save the output file
congested_links.to_csv('C:/Users/UserK/Desktop/flood_project/congested_links78.csv', index = False)
# road_link_nodes_2.to_csv('C:/Users/UserK/Desktop/flood_project/raods.csv', index = False)



# convex_hull_polygons.to_csv('D:/BILAL/5. shortest path/daejon network master data/convex_hull_polygons.csv', index = False)

# the optimal value of k is the elbow method, choose the inflection point in K-Means clustering.


# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(congested_links['O_x'],congested_links['O_y'], c= model.labels_)
# # sns.lmplot(data=congested_links, x='O_x', y='O_y', hue= model.labels_, fit_reg=False, legend=True, legend_out=True)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()

# #plotting without noises
# cong_filter = congested_links[congested_links['Clusters']!=-1]
# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(cong_filter['O_x'],cong_filter['O_y'],c=cong_filter.Clusters)
# # sns.lmplot(data=cong_filter, x='O_x', y='O_y', hue= 'Clusters', fit_reg=False, legend=True, legend_out=True)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()
# #cluster wise ranking
# congested_links['cl_links_repeatition'] = congested_links.groupby(['Clusters'])['links_repeatition'].transform('mean')
# congested_links['cl_road_cong'] = congested_links.groupby(['Clusters'])['road_cong'].transform('mean')
# congested_links['cl_links_count'] = congested_links.groupby(['Clusters'])['LINK_ID'].transform('count')

# #making convex hull around clusters #convex hull is not working that good as it needs three points atleast
# z=[] #HULL simplices coordinates will be appended here
# for i in range (0,num_clusters-1):
#     dfq=congested_links[congested_links['Clusters']==i]
#     Y = np.array(dfq[['O_y', 'O_x']])
#     hull = ConvexHull(Y)
#     plt.plot(Y[:, 1],Y[:, 0],  'o')
#     z.append(Y[hull.vertices,:].tolist())
#     for simplex in hull.simplices:
#         ploted=plt.plot( Y[simplex, 1], Y[simplex, 0],'k-',c='m')
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# plt.show()
# convex_hull_polygons = z


