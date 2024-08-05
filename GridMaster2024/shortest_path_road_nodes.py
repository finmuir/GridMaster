import networkx as nx
import matplotlib.pyplot as plt
import math
def find_shortest_road_path(roadData, start_node,end_node):
    # Initialize a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for i in range(0, len(roadData) - 1, 2):
        start = (roadData[i]['x'], roadData[i]['y'])
        end = (roadData[i + 1]['x'], roadData[i + 1]['y'])
        G.add_edge(start, end)


    # Find the shortest path using Dijkstra's algorithm
    shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight', method='dijkstra')

    # Extract x and y coordinates for plotting
    path_x, path_y = zip(*shortest_path)
    print(path_x,
          path_y)
    total_dist = 0
    for i in range(len(path_x)-1):
        dist = calc_distance(path_x[i],path_y[i],path_x[i+1],path_y[i+1])
        total_dist = total_dist + dist


def calc_distance(x1,y1,x2, y2):
    # Radius of the Earth in meters
    R = 6371000

    # Converts latitude and longitude points  from degrees to radians
    lat1 = math.radians(y1)
    lon1 = math.radians(x1)
    lat2 = math.radians(y2)
    lon2 = math.radians(x2)

    # Haversine formula calculates distance between two point on sphere
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance



