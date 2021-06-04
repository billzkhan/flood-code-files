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
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import sklearn.cluster as cluster
import seaborn as sns
import time
from sklearn.neighbors import NearestNeighbors
from geopy.distance import great_circle
from shapely.geometry import Point, MultiPoint
from matplotlib import cm
from scipy.spatial import ConvexHull
from functools import reduce
import networkx as nx
import geopandas as gpd
import itertools
import copy
import osmnx as ox


#function to flatten the list of list to return one list #run this command in console
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

#loading speed files for flooding, medium and normal days per 15 min data
speed_1 = pd.read_csv('D:/BILAL/5. shortest path/daejon network master data/MRT_BASE_TF_INFO_15MN_20200730.csv')
speed_1 = speed_1.iloc[:,1:6]
speed_2 = pd.read_csv('D:/BILAL/5. shortest path/daejon network master data/MRT_BASE_TF_INFO_15MN_20200813.csv')
speed_2 = speed_2.iloc[:,1:6]
speed_12rse = pd.merge(speed_1,speed_2,on=['LINK_ID','HH_ID','MN_ID'], how='left')

#VDS speed data
speed_1v = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/VDS 5 min data/5.속도및교통량_5분(20200730).xlsx')
speed_1v.drop(index=0, inplace=True)
speed_1v = speed_1v.iloc[:,[1,2,3,7,8]]
speed_1v = speed_1v.rename({'VDS_TR_VOL_YMDH':'VDS_flood_V'},axis=1)
speed_2v = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/VDS 5 min data/6.속도및교통량_5분(20200813).xlsx')
speed_2v.drop(index=0, inplace=True)
speed_2v = speed_2v.iloc[:,[1,2,3,7,8]]
speed_2v = speed_2v.rename({'VDS_TR_VOL_YMDH':'VDS_normal_V'},axis=1)

#merging vds speeds and volume
speed_12v = pd.merge(speed_1v,speed_2v,on=['LINK_ID','HH_ID','MN_ID'], how='left')
#merging rse speeds and TT
speed = pd.merge(speed_12rse,speed_12v,on=['LINK_ID','HH_ID','MN_ID'], how='left')

#loading main master data which doesnt change
road_link = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/4.도로-링크매핑(d_road_route_link).xlsx')
road_link.drop(index=0, inplace=True)
road_link = road_link.iloc[:,0:4]
node = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/2.노드마스터(d_node).xlsx')
node.drop(index=0, inplace=True)
node['START_NODE_ID'] = node['NODE_ID']
node['END_NODE_ID'] = node['NODE_ID']

link = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/1.링크마스터(d_link).xlsx')
link.drop(index=0, inplace=True)
link = link.rename({'SEQ':'SEQ_link'}, axis = 1)
road = pd.read_excel('D:/BILAL/5. shortest path/daejon network master data/3.도로마스터(d_road_route).xlsx')
road.drop(index=0, inplace=True)

# road_link.sort_values(['ROAD_ROUTE_ID', 'SEQ'], ascending=[True, True], inplace=True)
road_link = pd.merge(road_link,road[['ROAD_ROUTE_ID','MAIN_ROAD_YN']], on = ['ROAD_ROUTE_ID'], how= 'left')
road_link = road_link.rename({'SEQ':'SEQ_route'},axis=1)
road_link_nodes = pd.merge(link[['LINK_ID', 'START_NODE_ID', 'END_NODE_ID', 'LINK_LEN','ROAD_NM','SEQ_link']],road_link,on='LINK_ID', how='left')
road_link_nodes_2 = pd.merge(road_link_nodes,node[['START_NODE_ID','POSX','POSY']],on='START_NODE_ID', how='left')
road_link_nodes_2 = pd.merge(road_link_nodes_2,node[['END_NODE_ID','POSX','POSY']],on='END_NODE_ID', how='left')

# #counting unbiques
# items = Counter(road.ROUTE_NM).keys()
# len(items)

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
#sorting
# flood.sort_values(['ROUTE_DIR', 'SEQ_route','SEQ_link'], ascending=[True, True,True], inplace=True)

# #snake formation in various time periods
# x=0
# snakes = {} #dictionary to make number of lists
# for i in range(0,8):
#     if(x>45):
#         break
#     else:
#         sample = flood[(flood['HH_ID']==8) & (flood['MN_ID']==x) & (flood['ROUTE_DIR']==3502)]
#         congested_links = sample[(sample['diff']<=-1) & (sample['TRVL_SPD_x']<=15) & (sample['TRVL_SPD_y']>=15)]
#         data = congested_links['SEQ_link']
#         data = data.to_list()
#         snakes["snake{0}".format(x)] = []
#         for k, g in groupby(enumerate(data), lambda i_x: i_x[0] - i_x[1]):
#             snakes["snake{0}".format(x)].append(list(map(itemgetter(1), g)))
#         x = x+15

# #snake formation at the end of time period
# # x1 = list(snakes.values())[0]
# total_snakes = flatten(list(snakes.values()))
# #snakes.keys()
# snakes_4 = []
# for k, g in groupby(enumerate(total_snakes), lambda i_x: i_x[0] - i_x[1]): snakes_4.append(list(map(itemgetter(1), g)))
# snakes_4.sort()
# actual_snake = list(snakes_4 for snakes_4,_ in itertools.groupby(snakes_4))
# #total number of snake with links more than one in 1 hr time span
# more = 0
# b = []
# for elem in actual_snake:
#     if(len(elem)>1):
#         more=more+1
#         b.append(elem)
# c = flatten(b)

#plotting of speed data of flood and normal
plt.hist(flood['TRVL_SPD_x'])
plt.ylim(0, 800000)

plt.hist(flood['TRVL_SPD_y'])
plt.ylim(0, 800000)

#plotting.................
#Clustering
#sample
sample = flood[((flood['HH_ID']==7) | (flood['HH_ID']==8)) & (flood['ROUTE_DIR']==3502)]
sample_2 = sample.drop_duplicates(subset=['LINK_ID'])
sample_2['links_in_route'] = sample_2.groupby(['ROAD_ROUTE_ID'])['LINK_ID'].transform('count')
sample_2 = sample_2[['ROAD_ROUTE_ID','LINK_ID','links_in_route']]
#congested links
# congested_links = sample[(sample['diff']<=-3) & (sample['TRVL_SPD_x']<=10) & (sample['TRVL_SPD_y']>=10)]
congested_links = sample[(sample['diff']<=-10) & (sample['TRVL_SPD_x']!=30)]
congested_links['links_repeatition'] = congested_links['LINK_ID'].map(congested_links['LINK_ID'].value_counts()) #how many times each link is repeated in different time intervals
#remove the duplicated links on time interval
congested_links = congested_links.drop_duplicates(subset=['LINK_ID'])
congested_links['cong_links'] = congested_links.groupby(['ROAD_ROUTE_ID'])['LINK_ID'].transform('count')
#merge congested links and original sample
congested_links = congested_links.merge(sample_2.drop_duplicates(subset=['LINK_ID']), how='left')
congested_links = congested_links.sort_values('ROAD_ROUTE_ID')
congested_links['road_cong'] =  (congested_links['cong_links']/congested_links['links_in_route'])*100

coords = congested_links[['O_y', 'O_x']].to_numpy()

#nearest neighbours to find good epsilon value
# Calculate the average distance between each point in the data set and its 20 nearest neighbors (my selected MinPts value).
#Elbow method
neighbors = NearestNeighbors(n_neighbors=3)
neighbors_fit = neighbors.fit(np.radians(congested_links[['O_y','O_x']]))
distances, indices = neighbors_fit.kneighbors(np.radians(congested_links[['O_y','O_x']]))
# Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances*6371.008)

#dbscan calculation.............
kms_per_radian = 6371.0088
best_val = 0.5
# rec_dist = best_val*kms_per_radian
rec_dist = best_val
#DBSCAN algorithm................
dist = rec_dist # In KM
epsilon = dist / kms_per_radian
epsilon = dist / kms_per_radian
#real attempt at dbscan
model = DBSCAN(eps = epsilon, min_samples=3, algorithm='ball_tree', metric='haversine')
model.fit(np.radians(congested_links[['O_y','O_x']])) #converting lat lon to radians as haversine takes values in radians
cluster_labels = model.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
clusters = clusters.drop(clusters.index[[len(clusters)-1]])
print('Number of clusters: {}'.format(num_clusters))
congested_links['Clusters'] = model.labels_
congested_links['Clusters'].value_counts()

fig, ax = plt.subplots(figsize=(6,5))
ox.plot_graph(ox.graph_from_place('Daejeon, South Korea'))
ax.scatter(congested_links['O_x'],congested_links['O_y'], c= model.labels_)
# sns.lmplot(data=congested_links, x='O_x', y='O_y', hue= model.labels_, fit_reg=False, legend=True, legend_out=True)
plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
fig.show()

#plotting without noises
cong_filter = congested_links[congested_links['Clusters']!=-1]
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(cong_filter['O_x'],cong_filter['O_y'],c=cong_filter.Clusters)
# sns.lmplot(data=cong_filter, x='O_x', y='O_y', hue= 'Clusters', fit_reg=False, legend=True, legend_out=True)
plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
fig.show()
#cluster wise ranking
congested_links['cl_links_repeatition'] = congested_links.groupby(['Clusters'])['links_repeatition'].transform('mean')
congested_links['cl_road_cong'] = congested_links.groupby(['Clusters'])['road_cong'].transform('mean')

#Cluster of clusters..................................................................................
#trying to reduce the number of clusters formed by combining closer clusters together
# def get_centermost_point(cluster):
#     centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
#     centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
#     return tuple(centermost_point)
# centermost_points = clusters.map(get_centermost_point)
# lats, lons = zip(*centermost_points)
# rep_points = pd.DataFrame({'lon':lons, 'lat':lats})

# #finding optimal distance per node
# neighbors = NearestNeighbors(n_neighbors=2)
# neighbors_fit = neighbors.fit(np.radians(rep_points[['lat','lon']]))
# distances, indices = neighbors_fit.kneighbors(np.radians(rep_points[['lat','lon']]))
# # Sort distance values by ascending value and plot
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances*6371.0088)
# # Points sorted by distance to the 20th nearest neighbor,This concept of diminishing returns applies here because while increasing the number of clusters will always improve the fit of the model, it also increases the risk that overfitting will occur.

# #dbscan calculation.............
# kms_per_radian = 6371.0088
# best_val = 1
# rec_dist = best_val
# #DBSCAN algorithm................
# dist = rec_dist # In KM
# epsilon = dist / kms_per_radian
# #real attempt at dbscan
# model = DBSCAN(eps = epsilon, min_samples=2, algorithm='ball_tree', metric='haversine')
# model.fit(np.radians(rep_points[['lat','lon']])) #converting lat lon to radians as haversine takes values in radians
# cluster_labels = model.labels_
# num_clusters = len(set(cluster_labels))

# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(rep_points['lon'],rep_points['lat'], c= model.labels_)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()
# rep_points['Clusters_2'] = model.labels_
# rep_points['Clusters_2'].value_counts()

# #plotting without noises
# cong_filter = rep_points[rep_points['Clusters_2']!=-1]
# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(cong_filter['lon'],cong_filter['lat'],c=cong_filter.Clusters_2)
# # sns.lmplot(data=cong_filter, x='lon', y='lat', hue='Clusters_2', fit_reg=False, legend=True, legend_out=True)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()

# #merging the cluster and cluster of clusters to obtain actual clusters
# rep_points['index1'] = rep_points.index
# congested_links = pd.merge(congested_links, rep_points[['index1','Clusters_2']], how='left', left_on='Clusters',right_on='index1')
# congested_links['Clusters_ac'] = 1
# for i in range(0,len(congested_links)):
#     if(congested_links['Clusters_2'][i]==-1):
#         congested_links['Clusters_ac'][i] = congested_links['Clusters'][i]
#     elif(congested_links['Clusters_2'][i]==congested_links['Clusters'][i]):
#         congested_links['Clusters_ac'][i] = congested_links['Clusters'][i]
#     elif(congested_links['Clusters_2'][i]!=congested_links['Clusters'][i]):
#         congested_links['Clusters_ac'][i] = congested_links['Clusters_2'][i]+1000
# congested_links.loc[congested_links.Clusters == -1, 'Clusters_ac'] = -1
  


# #plotting
# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(congested_links['O_x'],congested_links['O_y'], c= congested_links['Clusters_ac'])
# # sns.lmplot(data=congested_links, x='O_x', y='O_y', hue= 'Clusters_ac', fit_reg=False, legend=True, legend_out=True)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()

# #plotting without noises
# cong_filter = congested_links[congested_links['Clusters_ac']!=-1]
# fig, ax = plt.subplots(figsize=(6,5))
# ax.scatter(cong_filter['O_x'],cong_filter['O_y'],c=cong_filter.Clusters_ac)
# # sns.lmplot(data=cong_filter, x='O_x', y='O_y', hue= 'Clusters_ac', fit_reg=False, legend=True, legend_out=True)
# plt.xlim([min(congested_links['O_x'])+0.01,max(congested_links['O_x'])+0.01])
# plt.ylim([min(congested_links['O_y'])+0.01,max(congested_links['O_y'])+0.01])
# fig.show()

#K means clustering algorithm......................not well suited on geo data
# cost = []
# for i in range(1,11):
#     kmeans = cluster.KMeans(n_clusters = i, init = 'k-means++', max_iter = 500)
#     kmeans = kmeans.fit(np.radians(df[['y','x']]))
#     cost.append(kmeans.inertia_)

# # plot the cost against K values 
# plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
# plt.xlabel("Value of K") 
# plt.ylabel("Sqaured Error (Cost)") 
# plt.show() # clear the plot 
# # plotting the best
# kmeans = cluster.KMeans(n_clusters = 6, init = 'k-means++')
# kmeans = kmeans.fit(np.radians(df[['y','x']]))
# kmeans.cluster_centers_
# df['Clusters'] = kmeans.labels_
# df.head(10)
# df['Clusters'].value_counts()
# sns.scatterplot(x = 'y', y = 'x', hue = 'Clusters', data = df)


#HDBSCAN algorithm..........................
# import hdbscan
# clusterer = hdbscan.HDBSCAN()
# clusterer.fit(df[['y','x']])
# HDBSCAN* is basically a DBSCAN implementation for varying epsilon values
# and therefore only needs the minimum cluster size as single input parameter, but we need to specify the radius and number of links

#taking average per route
# avg_speed_x = flood.groupby(['ROAD_ROUTE_ID'])['TRVL_SPD_x'].mean()
# flood = pd.merge(flood,avg_speed_x,on = 'ROAD_ROUTE_ID', how = 'left')
# avg_speed_y = flood.groupby(['ROAD_ROUTE_ID'])['TRVL_SPD_y'].mean()
# flood = pd.merge(flood,avg_speed_y,on = 'ROAD_ROUTE_ID', how = 'left')
# flood['diff_route'] = flood['TRVL_SPD_x_y']-flood['TRVL_SPD_y_y']

#convert continuous to discrete variable
# flood_normal['TRVL_SPD_xgr'] = pd.cut(flood_normal['TRVL_SPD_x'],bins=[0,10,15,20,200], labels=['a','b','c','d'])

#filters
# cd1 = (flood_normal['ROUTE_DIR_x']==3501)
# cd2 = (flood_normal['MAIN_ROAD_YN']=='Y')
# cd3 = ((flood_normal['HH_ID']==7 )) 
# cd4 = (flood_normal['TRVL_SPD_x']<=15) 
# cd5 = (flood_normal['TRVL_SPD_y']>15)
# cd6 = (flood_normal['diff']>-5)

#test data after applying the arc failure conditions
# f_ntest = flood_normal.loc[cd2&cd3&cd4&cd5&cd6]

#removing route with one link congested and comes once too in two hours
# route_counts = f_ntest.ROAD_ROUTE_ID.value_counts()
# route_counts_1 = route_counts[f_ntest.ROAD_ROUTE_ID.value_counts()!=1].index.tolist()
# f_ntest2 = f_ntest[f_ntest['ROAD_ROUTE_ID'].isin(route_counts_1)]

#time consideration, if a arc failure happens during various timings, not just 15 min time span...condition needs to be applied?/
#remove the arcs that does not lie more than once
# link_counts = f_ntest2.LINK_ID.value_counts()
# link_counts_1 = link_counts[f_ntest2.LINK_ID.value_counts()!=1].index.tolist()
# f_ntest3 = f_ntest2[f_ntest2['LINK_ID'].isin(link_counts_1)]

#save the output file
flood.to_csv('D:/BILAL/5. shortest path/daejon network master data/flood_1.csv', index = False)
#plotting the network data and check the speed variation across various links

#spatial network modeling attempt
# eps_rad = 500 / 3671000. #meters to radians
# db = DBSCAN(eps=eps_rad, min_samples=3, metric='haversine', algorithm='ball_tree')
# congested_links['spatial_cluster'] = db.fit_predict(np.deg2rad(congested_links[['O_y', 'O_x']]))
# congested_links['nn'] = ox.get_nearest_nodes(g, X=congested_links['O_x'], Y=congested_links['O_y'], method='balltree')
# graph = ox.graph_from_place('Daejeon, South Korea')
# congested_links['nn'] = ox.get_nearest_nodes(graph, X=congested_links['O_x'], Y=congested_links['O_y'], method='balltree')

# nodes_unique = pd.Series(congested_links['nn'].unique())
# nodes_unique.index = nodes_unique.values
# #Attempt at network spatial clustering
# def network_distance_matrix(u, graph, vs=nodes_unique):
#     dists = [nx.dijkstra_path_length(graph, source=u, target=v, weight='length') for v in vs]
#     return pd.Series(dists, index=vs)
# node_dm = nodes_unique.apply(network_distance_matrix, graph=graph)
# node_dm = node_dm.astype(int)
# # reindex to create establishment-based net dist matrix
# ndm = node_dm.reindex(index=df['nn'], columns=df['nn'])

#converting our data into graph data.......................................
#making edge and node list
edgelist = road_link_nodes_2[['START_NODE_ID','END_NODE_ID','LINK_LEN']]
edgelist = list(zip(edgelist.START_NODE_ID, edgelist.END_NODE_ID,edgelist.LINK_LEN))
nodelist = road_link_nodes_2[['LINK_ID','O_x','O_y']]
# pos = nodelist.set_index('LINK_ID').T.to_dict('tuple')

g = nx.Graph()
# Add edges and edge attributes
for start, end, length in edgelist:
    # You can attach any attributes you want when adding the edge
    g.add_edge(start, end, length=length)
# # Add node attributes, adding nodes separately is not necessary but since we have node attributes, thats why
# for i, nlrow in nodelist.iterrows():
#     g.node[nlrow['LINK_ID']] = nlrow[1:].to_dict()
nx.info(g)
g.nodes()
len(g.nodes())
nx.draw(g, node_size=5)
# nx.draw_networkx

#printing out the edge with its attributes
# for node1, node2, data in g.edges(data=True):
#     print(data['length'])

#Network Analysis.....................................................
#Degree of Centrality
# #1.degree of connection
# node_deg = nx.degree(g)
# nx.degree(g, 2930257400)
# #2. most influential
# most_influential = nx.degree_centrality(g)
# sorted(nx.degree_centrality(g))
# for w in sorted(most_influential, key = most_influential.get, reverse=True):
#     print(w,most_influential[w])
# #3. Most important connection
# nx.eigenvector_centrality(g, max_iter=600, weight = 'LINK_LEN')
# most_imp_link = nx.eigenvector_centrality(g, max_iter=600)
# for w in sorted(most_imp_link, key = most_imp_link.get, reverse=True):
#     print(w,most_imp_link[w])
# #shortest path
# nx.shortest_path(g,1870014700, 1870013600)
# #4. between centrality
betweenness_node = nx.betweenness_centrality(g, weight = 'LINK_LEN')
# betweenness_edge = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
# for w in sorted(best_connector, key = best_connector.get, reverse=True):
#     print(w,best_connector[w])
#plotting with colors
pos = nx.spring_layout(g)
node_color = [20000.0*g.degree(v) for v in g]
node_size = [v*10000 for v in betweenness_node.values()]
plt.figure(figsize = (20,20))
nx.draw_networkx(g, pos = pos, with_labels=False, node_color=node_color, node_size = 5)
plt.axis('off')


df = pd.DataFrame(dict(
    #number of nodes connected to the node
    DEGREE = dict(g.degree),
    #nodes connected vs totlal possible connections
    DEGREE_CENTRALITY = nx.degree_centrality(g),
    #most important connection
    EIGENVECTOR = nx.eigenvector_centrality(g,max_iter=600, weight = 'LINK_LEN'),
    PAGERANK = nx.pagerank(g, weight = 'LINK_LEN'),
    #how close the other nodes are
    # Closeness centrality [1] of a node u is the reciprocal of the sum of the shortest path distances from u to all n-1 other nodes.
    CLOSENESS_CENTRALITY = nx.closeness_centrality(g, distance = 'LINK_LEN'),
    #how many times same node is traversed on shortest path
    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(g, weight = 'LINK_LEN'),
    # BETWEENNESS_EDGE = nx.edge_betweenness_centrality(g, weight = 'LINK_LEN')
))
df['START_NODE_ID'] = df.index
df2 = pd.merge(congested_links,df, how='left', on= 'START_NODE_ID')
df2 = pd.merge(congested_links,df, how='left', on= 'START_NODE_ID')
df2['cl_link_count'] = df2.Clusters.map(df2.Clusters.value_counts())
df2.to_csv('D:/BILAL/5. shortest path/daejon network master data/df3.csv', index = False)
df2['cl_betweenness'] = df2.groupby(['Clusters'])['BETWEENNESS_CENTRALITY'].transform('mean')
df2['cl_degreeC'] = df2.groupby(['Clusters'])['DEGREE_CENTRALITY'].transform('mean')


#Building a subgroup
#retrieve all nodes connected to given node
#method1
# group1 = nx.bfs_tree(g, 1870014700)
# group2 = nx.bfs_tree(g, 1870013600)
# group3 = nx.bfs_tree(g, 2920009302)
# nx.draw(group1, node_size=10)
# nx.draw(group2, node_size=10)
# #method2
# nx.node_connected_component(g,1870014700)



#Attempt at network optimization

# #converting our data into graph data.......................................
# #duplicated values solution
# import inflect
# p = inflect.engine()
# #convert columns to strings
# road_link_nodes_2['LINK_ID']= road_link_nodes_2['LINK_ID'].astype(str)
# road_link_nodes_2['START_NODE_ID']= road_link_nodes_2['START_NODE_ID'].astype(str)
# road_link_nodes_2['END_NODE_ID']= road_link_nodes_2['END_NODE_ID'].astype(str)
# #changing names of duplicated links and nodes
# road_link_nodes_2['LINK_ID'] += road_link_nodes_2.groupby('LINK_ID').cumcount().add(1).map(p.ordinal).radd('_')
# road_link_nodes_2['START_NODE_ID'] += road_link_nodes_2.groupby('START_NODE_ID').cumcount().add(1).map(p.ordinal).radd('_')
# road_link_nodes_2['END_NODE_ID'] += road_link_nodes_2.groupby('END_NODE_ID').cumcount().add(1).map(p.ordinal).radd('_')
# #making edge and node list
# edgelist = road_link_nodes_2[['START_NODE_ID','END_NODE_ID','LINK_LEN']]
# edgelist = list(zip(edgelist.START_NODE_ID, edgelist.END_NODE_ID,edgelist.LINK_LEN))
# nodelist = road_link_nodes_2[['LINK_ID','O_x','O_y']]
# node_attr = nodelist.set_index('LINK_ID').to_dict('index')
# g = nx.Graph()
# #Add nodes and node attributes
# nx.set_node_attributes(g, node_attr)
# # Add edges and edge attributes
# for start, end, length in edgelist:
#     # You can attach any attributes you want when adding the edge
#     g.add_edge(start, end, length=length)
# # # Add node attributes, adding nodes separately is not necessary but since we have node attributes, thats why
# # for i, nlrow in nodelist.iterrows():
# #     g.node[nlrow['LINK_ID']] = nlrow[1:].to_dict()
# nx.info(g)
# g.nodes()
# len(g.nodes())
# nx.draw(g, node_size=5)
# # nx.draw_networkx
# https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial#comments
















