import numpy as np
import shortest_path_road_nodes
# Haversine distance calculation
# def dist_calc(customer: dict, road: dict) -> float:
#     """
#     Calculates the distance between a customer and a roadnode using the Haversine formula.
#     """
#     # Extract coordinates
#     X = customer['x']
#     Y = customer['y']
#     X_c = road['x']
#     Y_c = road['y']
#
#     # Radius of the Earth in meters
#     R = 6371000
#
#     # Convert latitude and longitude from degrees to radians
#     lat1 = np.radians(Y)
#     lon1 = np.radians(X)
#     lat2 = np.radians(Y_c)
#     lon2 = np.radians(X_c)
#
#     # Haversine formula to calculate the distance between two points on a sphere
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     distance = R * c
#
#     return distance
#
#
#
# # Function to find the closest roadnode for each customer
# def find_closest_roadnode(customerData, roadData):
#     """
#     Finds the closest roadnode for each customer.
#     """
#     print('starting closest roadnode ')
#     closest_roadnodes = []
#
#
#     for Cluster in customerData:
#         if 'Type' in Cluster and Cluster['Type'] == 'customer_pole':
#             min_distance = float('inf')
#             closest_node = None
#
#
#             for roadnode in roadData:
#
#                 if 'Type' in roadnode and roadnode['Type'] == 'road_node':
#                     try:
#
#                         distance = dist_calc(Cluster, roadnode)
#
#
#
#                         if not isinstance(distance, (float, int)):
#                             print(f"Invalid distance value: {distance}")
#                             continue
#
#                         if distance < min_distance:
#                             min_distance = distance
#                             closest_node = roadnode
#                     except Exception as e:
#                         print(f"Error calculating distance for customer {Cluster} and roadnode {roadnode}: {e}")
#
#             if closest_node is not None:
#                 closest_roadnodes.append({
#                     'Cluster': Cluster,
#                     'closest_roadnode': closest_node,
#                     'distance': min_distance
#                 })
#
#
#     return closest_roadnodes


def build_paths(roadData, clusterData, source_coords):
    road_paths = []
    path_distances = []

    for cluster in clusterData:
        if 'Type' in cluster and cluster['Type'] == 'customer_pole':
            coords = (cluster['x'], cluster['y'])
            # Ensure coordinates are unpacked correctly
            source_lat, source_lon = source_coords
            dest_lat = float(coords[0])
            dest_lon = float(coords[1])
            print(dest_lat, type(dest_lon))

            # Call the function and unpack all three return values
            G, shortest_path, total_dist = shortest_path_road_nodes.find_shortest_road_path(
                roadData, (source_lat, source_lon), (dest_lat, dest_lon)
            )


            road_paths.append(shortest_path)
            path_distances.append(total_dist)

    print('Paths:', road_paths)
    print('Path distances:', path_distances)
    return road_paths, path_distances
