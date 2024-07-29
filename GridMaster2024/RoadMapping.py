import overpy
import pandas as pd

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

import pandas as pd

def format_road_data(result):
    nodes = []
    edges = []

    for way in result.ways:
        # Keep track of nodes and edges
        prev_node = None
        for i, node in enumerate(way.nodes):
            nodes.append({'lat': node.lat, 'lon': node.lon, 'type': 'road_node'})
            if prev_node:
                # Avoid connecting the last node to the first node if not a valid loop
                if not (i == len(way.nodes) - 1 and way.nodes[0] == node):
                    edges.append({'start': prev_node, 'end': (node.lat, node.lon)})
            prev_node = (node.lat, node.lon)

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    return nodes_df, edges_df

