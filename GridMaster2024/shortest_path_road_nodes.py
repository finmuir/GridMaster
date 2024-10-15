import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np


def find_shortest_road_path(roadData, start_node, end_node):
    'find the shortest path using road nodes to connect two points'
    G = nx.Graph()

    # Add nodes and edges to the graph with distances
    for i in range(len(roadData) - 1):
        if 'Type' in roadData[i] and roadData[i]['Type'] == 'road_node':
            if 'Type' in roadData[i+1] and roadData[i+1]['Type'] == 'road_node':
                start = (roadData[i]['x'], roadData[i]['y'])  # Lon, Lat
                end = (roadData[i + 1]['x'], roadData[i + 1]['y'])  # Lon, Lat
                dist = calc_distance(start[1], start[0], end[1], end[0])  # Passing Lat, Lon
                G.add_edge(start, end, weight=dist)

    # Ensure start_node and end_node are properly formatted tuples (Lon, Lat)
    start_node = tuple(start_node)
    end_node = tuple(end_node)

    # Add start and end nodes to the graph
    G.add_node(start_node)
    G.add_node(end_node)

    # Find the nearest nodes in roadData to start_node and end_node
    roadnodes = []
    for i in range(len(roadData)):
        if 'Type' in roadData[i] and roadData[i]['Type'] == 'road_node':
            roadnodes.append(roadData[i])

    start_nearest = min(roadnodes, key=lambda node: calc_distance(start_node[1], start_node[0], node['y'], node['x']))
    end_nearest = min(roadnodes, key=lambda node: calc_distance(end_node[1], end_node[0], node['y'], node['x']))

    start_nearest = (start_nearest['x'], start_nearest['y'])  # Lon, Lat
    end_nearest = (end_nearest['x'], end_nearest['y'])  # Lon, Lat

    # Connect start and end nodes to their nearest nodes in roadData
    G.add_edge(start_node, start_nearest, weight=calc_distance(start_node[1], start_node[0], start_nearest[1], start_nearest[0]))
    G.add_edge(end_node, end_nearest, weight=calc_distance(end_node[1], end_node[0], end_nearest[1], end_nearest[0]))

    # Find the shortest path using Dijkstra's algorithm
    try:
        shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight', method='dijkstra')
    except nx.NetworkXNoPath:
        print("No path found between the specified nodes.")
        return G, [], float('inf')

    # Extract lon and lat coordinates for plotting
    path_lon, path_lat = zip(*shortest_path)

    # Calculate total distance
    total_dist = 0
    for i in range(len(path_lon) - 1):
        dist = calc_distance(path_lat[i], path_lon[i], path_lat[i + 1], path_lon[i + 1])
        total_dist += dist

    # Return the graph, shortest path as (lon, lat) tuples, and total distance
    return G, shortest_path, total_dist

def calc_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000

    # Converts latitude and longitude points from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Haversine formula calculates distance between two points on the sphere
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

def find_closest_node_on_path(shortest_path, third_node):
    closest_node = None
    min_distance = float('inf')

    for path_node in shortest_path:
        dist = calc_distance(third_node[1], third_node[0], path_node[1], path_node[0])
        if dist < min_distance:
            min_distance = dist
            closest_node = path_node

    return closest_node, min_distance

def find_shortest_path_to_third_node(roadData, shortest_path, third_node):
    min_additional_path = None
    min_additional_dist = float('inf')

    # Find shortest path from each node on the original path to the third node
    for node in shortest_path:
        # Ensure start_node is a tuple
        start_node = tuple(node)
        # Find the shortest path from the current start_node to the third_node
        _, additional_path, additional_dist = find_shortest_road_path(roadData, start_node, third_node)

        if additional_dist < min_additional_dist:
            min_additional_dist = additional_dist
            min_additional_path = additional_path

    if min_additional_path is None:
        print("No path found to the third node.")
        return [], float('inf')

    return min_additional_path, min_additional_dist

def plot_graph(G, final_path):
    # Ensure all nodes in the final path are in the graph
    for node in final_path:
        if node not in G.nodes:
            G.add_node(node)

    # Prepare positions for nodes
    pos = {node: node for node in G.nodes}

    # Draw the graph
    nx.draw(G, pos, node_size=50, node_color='blue', edge_color='grey')

    # Draw the shortest path edges
    path_edges = list(zip(final_path, final_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    plt.show()

def thirdpoint_plot_graph(G, final_path, additional_path, third_node):
    # Ensure all nodes in the final path are in the graph
    for node in final_path:
        if node not in G.nodes:
            G.add_node(node)

    # Ensure the third node is in the graph
    if third_node not in G.nodes:
        G.add_node(third_node)

    # Prepare positions for nodes
    pos = {node: node for node in G.nodes}

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_size=50, node_color='blue', edge_color='grey')

    # Draw the shortest path edges
    path_edges = list(zip(final_path, final_path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    # Draw the additional path edges
    if additional_path:
        additional_path_edges = list(zip(additional_path, additional_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=additional_path_edges, edge_color='green', width=2)

    # Draw the third node in green
    nx.draw_networkx_nodes(G, pos, nodelist=[third_node], node_color='green', node_size=100)

    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

    plt.show()





# # Example road data
# roadData = [
#             {'x': 34.6057896, 'y': -14.2469046, 'Type': 'road_node'}, {'x': 34.6057226, 'y': -14.2469748, 'Type': 'road_node'},
#             {'x': 34.6056314, 'y': -14.2470196, 'Type': 'road_node'}, {'x': 34.6055549, 'y': -14.2470391, 'Type': 'road_node'},
#             {'x': 34.6054423, 'y': -14.2469995, 'Type': 'road_node'},{'x': 34.6050292, 'y': -14.2467941, 'Type': 'road_node'},
#             {'x': 34.6043479, 'y': -14.2465393, 'Type': 'road_node'}, {'x': 34.6031463, 'y': -14.2460402, 'Type': 'road_node'},
#             {'x':34.6019876, 'y': -14.2456034, 'Type': 'road_node'}, {'x': 34.5999129, 'y': -14.2449275, 'Type': 'road_node'},
#             {'x': 34.599532, 'y': -14.2447858, 'Type': 'road_node'}, {'x': 34.5991042, 'y': -14.2446389, 'Type': 'road_node'},
#             {'x': 34.598152, 'y': -14.2443009, 'Type': 'road_node'}, {'x': 34.5975968, 'y': -14.2441034, 'Type': 'road_node'},
#             {'x': 34.5970175, 'y': -14.2438824, 'Type': 'road_node'}, {'x': 34.5965749, 'y': -14.2437134, 'Type': 'road_node'}, {'x': 34.5961739, 'y': -14.2435691, 'Type': 'road_node'},
#             {'x': 34.5956683, 'y': -14.2433754, 'Type': 'road_node'}, {'x': 34.5950353, 'y': -14.2431233, 'Type': 'road_node'},
#             {'x': 34.5948127,'y': -14.2430323,'Type': 'road_node'}, {
#             'x': 34.5944868, 'y': -14.2429101, 'Type': 'road_node'}, {'x': 34.5939571, 'y': -14.2427073, 'Type': 'road_node'},
#             {'x': 34.5934448,'y': -14.2424993,'Type': 'road_node'}, {
#             'x': 34.5925865, 'y': -14.2421769, 'Type': 'road_node'}, {'x': 34.5916316, 'y': -14.2418026, 'Type': 'road_node'},
#             {'x': 34.5913419, 'y': -14.2416882, 'Type': 'road_node'}, {'x': 34.5910469, 'y': -14.2415634, 'Type':'road_node'},
#             {'x': 34.5908658, 'y': -14.2414789,'Type': 'road_node'}, {'x': 34.5906499, 'y': -14.2413736, 'Type': 'road_node'},
#             {'x': 34.5904192, 'y': -14.2412514, 'Type': 'road_node'}, {'x': 34.5899498, 'y': -14.2409524, 'Type': 'road_node'},
#             {'x': 34.589577, 'y': -14.2407029, 'Type': 'road_node'}, {'x': 34.5894764, 'y': -14.2406197, 'Type': 'road_node'}
#             , {'x': 34.589396, 'y': -14.2405417, 'Type': 'road_node'}, {'x': 34.5892712, 'y': -14.2404091,'Type': 'road_node'},
#             {'x': 34.5891304, 'y': -14.2402726,'Type': 'road_node'}, {'x':34.5890513,'y': -14.2401823,'Type': 'road_node'},
#             {'x': 34.5890084, 'y': -14.2401218, 'Type': 'road_node'}, {'x': 34.5889715, 'y': -14.2400789, 'Type': 'road_node'},
#             {'x': 34.5889266, 'y': -14.2400399, 'Type': 'road_node'}, {'x': 34.5888837, 'y': -14.2400113, 'Type': 'road_node'},
#             {'x': 34.5887884, 'y': -14.2399736, 'Type': 'road_node'}, {'x': 34.5886624,'y': -14.2399411, 'Type': 'road_node'},
#             {'x': 34.5885001,'y': -14.2399099,'Type': 'road_node'}, {
#             'x': 34.5882721, 'y': -14.2398748, 'Type': 'road_node'},{'x': 34.5881273, 'y':-14.2398436, 'Type': 'road_node'},
#             {'x': 34.5879731, 'y': -14.2397812, 'Type': 'road_node'},
#         ]
# #
# #
# # Source and destination coordinates
# source_coords = ( 34.60600833,-14.24580667,)  # Note: ( longitude.latitude)
# end_node_coords = (34.58967970133, -14.242936167)
#
# # Create the graph with road data and find the shortest path
# G, shortest_path, total_dist = find_shortest_road_path(roadData, source_coords, end_node_coords)
# print("Shortest Path:", shortest_path)
# print("Total Distance (meters):", total_dist)
# #
# # Plot the graph and the shortest path
# plot_graph(G, shortest_path)
# #
# # Third node coordinates (example)
# third_node_coords = (34.5890600835,-14.24180660)
# # Step 1: Find the shortest path between source and destination
# G, shortest_path, total_dist = find_shortest_road_path(roadData, source_coords, end_node_coords)
# print("Shortest Path without third node:", shortest_path)
# print("Total Distance (meters) without third node:", total_dist)
#
# closestnode, distance123 = find_closest_node_on_path(shortest_path, third_node_coords)
# print('closest node ', closestnode)
#
# # Step 2: Find the additional path to the third node from the closest point on the shortest path
# additional_path_to_third, additional_dist_to_third = find_shortest_path_to_third_node(roadData, shortest_path,
#                                                                                       third_node_coords)
# print("Additional Path to third node:", additional_path_to_third)
# print("Additional Distance (meters):", additional_dist_to_third)
#
# # Combine paths for final plotting (but only return the additional path and distance)
# if additional_path_to_third:
#     closest_node_on_path, _ = find_closest_node_on_path(shortest_path, third_node_coords)
#
#     # Insert the additional path at the correct place in the shortest path
#     index_of_closest = shortest_path.index(closest_node_on_path)
#     final_path = shortest_path[:index_of_closest + 1] + additional_path_to_third[1:]
# else:
#     final_path = shortest_path
#
# # Step 4: Plot the final path on the graph
# thirdpoint_plot_graph(G, shortest_path, additional_path_to_third, third_node_coords)
