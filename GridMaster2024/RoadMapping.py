import overpy
import pandas as pd
import networkx as nx

def calculate_bounding_box(center_lat, center_lon, distance_threshold):
    lat_offset = (distance_threshold*0.25) / 111111  # Approximate latitude degrees per meter
    lon_offset = distance_threshold*0.25 / (111111 * abs(center_lat))  # Approximate longitude degrees per meter
    north = center_lat + lat_offset
    south = center_lat - lat_offset
    east = center_lon + lon_offset
    west = center_lon - lon_offset
    return south, west, north, east

def find_roads(south, west, north, east):
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
    # Create a graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    nodes_data = []
    edges_data = []

    for way in result.ways:
        nodes = way.get_nodes(resolve_missing=True)
        for i in range(len(nodes)-1):
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
    # Create DataFrames for Plotly Express
    nodes_df = pd.DataFrame(nodes_data).drop_duplicates()
    edges_df = pd.DataFrame(edges_data)

    return nodes_df, edges_df

