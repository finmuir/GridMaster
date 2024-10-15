import overpy
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

import math


def calculate_bounding_box(center_lat, center_lon, distance_threshold):
    'creates a box around the source coordinates '
    # Approximate latitude degrees per meter
    lat_offset = (distance_threshold * 0.5) / 111111

    # Approximate longitude degrees per meter, accounting for the latitude
    lon_offset = (distance_threshold * 0.5) / (111111 * math.cos(math.radians(center_lat)))

    north = center_lat + lat_offset
    south = center_lat - lat_offset
    east = center_lon + lon_offset
    west = center_lon - lon_offset

    return south, west, north, east


def find_roads(south, west, north, east):
    'using open street map library all the roads within the bounding box are found'
    try:
        api = overpy.Overpass()
        query = f"""
        [out:json];
        (
          way["highway"]({south},{west},{north},{east});
          >;
        );
        out body;
        """
        result = api.query(query)
        return result
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def format_road_data(result):
    'formats the road data into a series of nodes and edges'
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    nodes_data = []
    edges_data = []

    for way in result.ways:
        nodes = way.get_nodes(resolve_missing=True)
        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            G.add_node(node1.id, lat=float(node1.lat), lon=float(node1.lon))
            G.add_node(node2.id, lat=float(node2.lat), lon=float(node2.lon))
            G.add_edge(node1.id, node2.id, way_id=way.id)
            edges_data.append({
                'lat': [float(node1.lat), float(node2.lat)],
                'lon': [float(node1.lon), float(node2.lon)]
            })
            nodes_data.extend([
                {'id': node1.id, 'lat': float(node1.lat), 'lon': float(node1.lon)},
                {'id': node2.id, 'lat': float(node2.lat), 'lon': float(node2.lon)}
            ])

    # Create DataFrames for Plotly
    nodes_df = pd.DataFrame(nodes_data).drop_duplicates()
    edges_df = pd.DataFrame(edges_data)


    return nodes_df, edges_df

'below is just a visulisation of the road data '
#
# def plot_road_network(nodes_df, edges_df):
#     # Create Plotly figure
#     fig = go.Figure()
#
#     # Add OpenStreetMap as base layer
#     fig.update_layout(
#         mapbox=dict(
#             style="open-street-map",
#             center=dict(lat=nodes_df['lat'].mean(), lon=nodes_df['lon'].mean()),
#             zoom=12
#         )
#     )
#
#     # Add edges to the plot
#     for _, edge in edges_df.iterrows():
#         fig.add_trace(go.Scattermapbox(
#             lon=edge['lon'],
#             lat=edge['lat'],
#             mode='lines',
#             line=dict(width=2, color='black'),
#             name='Roads'
#         ))
#
#     # Add nodes to the plot
#     fig.add_trace(go.Scattermapbox(
#         lon=nodes_df['lon'],
#         lat=nodes_df['lat'],
#         mode='markers',
#         marker=dict(size=5, color='red'),
#         name='Nodes'
#     ))
#
#     fig.update_layout(
#         title='Road Network Visualization',
#         showlegend=True
#     )
#
#     fig.show()
#
#
# # Example usage
# center_lat = -2.86707 # Latitude of Bugarama
# center_lon = 30.53309
# distance_threshold = 1000  # 5 km
#
# south, west, north, east = calculate_bounding_box(center_lat, center_lon, distance_threshold)
# result = find_roads(south, west, north, east)
# if result:
#     nodes_df, edges_df = format_road_data(result)
#     plot_road_network(nodes_df, edges_df)
