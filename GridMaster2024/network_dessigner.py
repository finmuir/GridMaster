import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

import shortest_path_road_nodes as s_path

class Node:

    def __init__(self, location, power_demand, node_id):
        """
        Represents any element in the network that draws power, such
        as customers and clusters of customers.

        Parameters
        ----------
        location : array-like
            1x2 array containing X and Y coordinates of source.
        power_demand : array-like
            Power demand of node.
        node_id : str, optional
            Node identifier. The default is None.

        """

        self.loc = tuple(location)  # [0] is X, [1] is Y
        self.node_id = str(node_id)
        self.Pdem = np.array(power_demand, dtype="float64")
        self.csrt_sat = True  # constraints satisfied upon creation

        # -------CONNECTIONS---------------------------------------------------#

        self.parent = 0  # all nodes initially connected to source
        self.children = []
        self.line_res = 0  # resistance in line between node and its parent

        # -------CURRENT/VOLTAGE ARRAYS----------------------------------------#

        self.I = 0  # current drawn by node at each hour
        self.I_line = 0  # current in line at each hour
        self.V = 0  # voltage across node at each time step

        # -------CMST TRACKERS-------------------------------------------------#

        self.V_checked = False
        self.I_checked = False

        print(f"Node created: ID={self.node_id}, Location={self.loc}, Power Demand={self.Pdem}")

    def isgate(self):
        """
        Returns gate status of node.

        Returns
        -------
        bool
            True if node is gate node. False otherwise.

        """

        if self.parent == 0:
            return True
        else:
            return False

    def has_children(self):
        """
        Returns parent status of node.

        Returns
        -------
        bool
            True if node has children. False otherwise.

        """

        if self.children != []:
            return True

        else:
            return False


class Source:
    node_id = "SOURCE"

    def __init__(self, location):
        self.loc = tuple(location)

        print(f"Source created: Location={self.loc}")

    def isgate(self):
        return False


class cluster_road_nodes:

    def __init__(self, location, power_demand, node_id, path, distance):
        """
        Represents any element in the network that draws power, such
        as customers and clusters of customers.

        Parameters
        ----------
        location : array-like
            1x2 array containing X and Y coordinates of source.
        power_demand : array-like
            Power demand of node.
        node_id : str, optional
            Node identifier. The default is None.

        """

        self.loc = tuple(location)  # [0] is X, [1] is Y
        self.node_id = str(node_id)
        self.Pdem = np.array(power_demand, dtype="float64")
        self.csrt_sat = True  # constraints satisfied upon creation

        # -------CONNECTIONS---------------------------------------------------#

        self.parent = 0  # all nodes initially connected to source
        self.children = []
        self.line_res = 0  # resistance in line between node and its parent

        # -------CURRENT/VOLTAGE ARRAYS----------------------------------------#

        self.I = 0  # current drawn by node at each hour
        self.I_line = 0  # current in line at each hour
        self.V = 0  # voltage across node at each time step

        # -------CMST TRACKERS-------------------------------------------------#

        self.V_checked = False
        self.I_checked = False

        # -------Road paths -------------------------------------------------#
        self.path=path
        self.distance =distance

        print(f"Cluster Road Node created: ID={self.node_id}, Location={self.loc}, "
              f"Power Demand={self.Pdem},  Distance={self.distance},")
        #Path = {self.path}


    def isgate(self):
        """
        Returns gate status of node.

        Returns
        -------
        bool
            True if node is gate node. False otherwise.

        """

        if self.parent == 0:
            return True
        else:
            return False

    def has_children(self):
        """
        Returns parent status of node.

        Returns
        -------
        bool
            True if node has children. False otherwise.

        """

        if self.children != []:
            return True

        else:
            return False


class NetworkDesigner:
    def __init__(self, road_data, source_location, node_locations, node_power_demand,
                 network_voltage, pole_spacing, res_per_km, max_current, max_V_drop,
                 scl=1, node_ids=None, V_reg=6):
        """
        Designs a network connecting nodes using Esau-Williams CMST.

        Parameters
        ----------
        road_data : list
            Road data for finding paths.
        source_location : tuple
            Coordinates (X, Y) of the source.
        node_locations : list of tuples
            List of node locations.
        node_power_demand : list of arrays
            List of power demands for each node.
        network_voltage : float
            Operating voltage of the network.
        pole_spacing : float
            Spacing between poles.
        res_per_km : float
            Resistance per kilometer of cable.
        max_current : float
            Maximum current rating.
        max_V_drop : float
            Maximum allowable voltage drop.
        scl : float, optional
            Scaling factor for coordinates. Default is 1.
        node_ids : list of str, optional
            Identifiers for nodes. Default is None.
        V_reg : float, optional
            Voltage regulation percentage. Default is 6.
        """
        if node_ids is None:
            node_ids = []

        self.nodes = [Source(source_location)]

        for idx in range(len(node_locations)):
            loc = node_locations[idx]
            node_id = node_ids[idx] if idx < len(node_ids) else None
            power_demand = node_power_demand[idx]
            self.nodes.append(Node(loc, power_demand, node_id))

            path, road_distance = s_path.find_shortest_road_path(road_data, source_location, loc)
            self.nodes.append(cluster_road_nodes(loc, power_demand, node_id, path, road_distance))

        self.Vnet = network_voltage
        self.pole_spacing = pole_spacing
        self.Vdrop_max = max_V_drop
        self.res_meter = res_per_km / 1000
        self.Imax = max_current

        print("Network Designer initialized.")
        print(f"Source: {self.nodes[0].loc}")
        for node in self.nodes[1:]:
            if isinstance(node, cluster_road_nodes):
                print(f"Cluster Road Node: ID={node.node_id}, Location={node.loc}, "
                      f"Power Demand={node.Pdem}, Distance={node.distance}")
            else:
                print(f"Node: ID={node.node_id}, Location={node.loc}, Power Demand={node.Pdem}")

    @classmethod
    def import_from_csv(cls, cluster_data, road_data, source_coord, network_voltage, pole_spacing,
                        res_per_km, max_current, scl=1, max_V_drop=0):
        """
        Creates a NetworkDesigner instance from CSV data.

        Parameters
        ----------
        cluster_data : list
            List of dictionaries containing cluster data.
        road_data : list
            Road data for finding paths.
        source_coord : tuple
            Coordinates (X, Y) of the source.
        network_voltage : float
            Operating voltage of the network.
        pole_spacing : float
            Spacing between poles.
        res_per_km : float
            Resistance per kilometer of cable.
        max_current : float
            Maximum current rating.
        scl : float, optional
            Scaling factor for coordinates. Default is 1.
        max_V_drop : float, optional
            Maximum allowable voltage drop. Default is 0.
        """
        node_locs = []
        power_demands = []
        node_ids = []

        for cluster in cluster_data:
            if cluster['Type'] == 'customer_pole':
                node_locs.append((cluster['x'], cluster['y']))
                power_demands.append(cluster['load'])
                node_ids.append(cluster['Cluster'])

        return cls(road_data, source_coord, node_locs, power_demands, network_voltage, pole_spacing,
                   res_per_km, max_current, max_V_drop, scl, node_ids)

def calc_dist(x,y,x2,y2):
    """
    Creates array populated with distances between customers and centroid
    of cluster. Array is 1D, shape 1 x len(customers).

    Returns
    -------
    Numpy array
        Array populated with internal centroid-customer distances.

    """

    # Radius of the Earth in meters
    R = 6371000

    # Converts latitude and longitude points  from degrees to radians
    lat1 = np.radians(y)
    lon1 = np.radians(x)
    lat2 = np.radians(y2)
    lon2 = np.radians(x2)

    # Haversine formula calculates distance between two point on sphere
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance



