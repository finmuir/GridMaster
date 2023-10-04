# # -*- coding: utf-8 -*-
# """
#     Illustrates CustomerClustering object with data from CSV file.
# """
#
#
# import random
#
# from MappingTest2 import customer_clustering as cc
#
# import matplotlib.pyplot as plt
#
# clusterer = cc.CustomerClustering.import_from_csv(
#     "csv_uploads/nodes_datapdem.csv",
#     network_voltage=230,
#     pole_cost=100,
#     pole_spacing=50,
#     resistance_per_km=4.61,
#     current_rating=37,
#     cost_per_km=1520,
#     max_voltage_drop=11.5
#     )
#
# clusterer.cluster(max_customers=6)
#
# # DRAWING
# plt.figure()
# for idx,cluster in enumerate(clusterer.clusters):
#     x_c = cluster.position[0]
#     y_c = cluster.position[1]
#     x = [customer.position[0] for customer in cluster.customers]
#     y = [customer.position[1] for customer in cluster.customers]
#     color = random.randint(0, 500)
#     if idx == 0:
#         plt.scatter(x,y,label="customers")  # customers
#         plt.scatter(x_c, y_c, c="black", marker="+", label="poles")  # type: ignore  # pole
#     else:
#         plt.scatter(x,y)  # customers
#         plt.scatter(x_c, y_c, c="black", marker="+")  # type: ignore  # pole
# plt.title("Customer Clustering")
# plt.legend()
# plt.show()
#
# print("\nnumber of clusters:",len(clusterer.clusters))
# print("\ntotal line cost:",clusterer.total_cost)

import random
import plotly.express as px
import pandas as pd

from MappingTest2 import customer_clustering as cc

clusterer = cc.CustomerClustering.import_from_csv(
    "csv_uploads/nodes_datapdem.csv",
    network_voltage=230,
    pole_cost=100,
    pole_spacing=50,
    resistance_per_km=4.61,
    current_rating=37,
    cost_per_km=1520,
    max_voltage_drop=11.5
    )

clusterer.cluster(max_customers=6)

# Preparing data
cluster_data = []
for idx, cluster in enumerate(clusterer.clusters):
    x_c = cluster.position[0]
    y_c = cluster.position[1]
    x = [customer.position[0] for customer in cluster.customers]
    y = [customer.position[1] for customer in cluster.customers]
    color = random.randint(0, 500)
    for i in range(len(x)):
        cluster_data.append({"x": x[i], "y": y[i], "Cluster": idx, "Type": "customer"})
    cluster_data.append({"x": x_c, "y": y_c, "Cluster": idx, "Type": "pole"})

df = pd.DataFrame(cluster_data)

# Plotting
fig = px.scatter(df, x='x', y='y', color='Cluster', symbol='Type',
                 title="Customer Clustering",
                 labels={"x": "", "y": ""})  # remove axis labels

fig.show()

print("\nnumber of clusters:", len(clusterer.clusters))
print("\ntotal line cost:", clusterer.total_cost)
