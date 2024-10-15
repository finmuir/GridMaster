import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import copy
import shortest_path_road_nodes as s_path


def distance_calc( x1, y1, x2, y2):
    # Radius of the Earth in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(y1)
    lon1 = math.radians(x1)
    lat2 = math.radians(y2)
    lon2 = math.radians(x2)

    # Haversine formula calculates distance between two points on a sphere
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

class Source:
    'used to create source instance node id = 0 and has location'
    node_id = 0

    def __init__(self, x, y):
        self.loc = (y, x)
        print(f"Source created: Location={self.loc}")

    def isgate(self):
        return False

class Nodes:

    def __init__(self, location, power_demand, node_id):
        """
        Represents clusters of customers.

        Parameters
        ----------
        location : array-like
            1x2 array containing X and Y coordinates of customer pole.
        power_demand : array-like
            Power demand of all customers in the cluster added together.
        node_id : str, optional
            Node identifier. The default is None.
        """

        self.loc = tuple(location)  # [0] is X, [1] is Y
        self.node_id = str(node_id)
        self.Pdem = np.array(power_demand, dtype="float64")
        self.csrt_sat = True  # constraints satisfied upon creation

        # -------CONNECTIONS---------------------------------------------------#
        #tracks the connections to its parent node
        self.subtree = None
        self.parent = 0  # All nodes initially connected to source
        self.parent_distance = 0  # Tracks distance to parent node
        self.parent_path = []  # Tracks the path to the parent node
        self.children = []
        self.line_res = 0

        # -------CURRENT/VOLTAGE ARRAYS----------------------------------------#

        self.I = 0  # current drawn by node at each hour
        self.I_line = 0  # current in line at each hour
        self.V = 0  # voltage across node at each time step

        # -------CMST TRACKERS-------------------------------------------------#

        self.V_checked = False
        self.I_checked = False

    def set_parent(self, parent_node, distance, path):
        """
        Sets the parent connection, distance, and path to the parent node.

        Parameters
        ----------
        parent_node : Nodes
            The parent node (initially source).
        distance : float
            Distance from node to its parent.
        path : list
            Path from this node to its parent.
        """
        self.parent = parent_node.node_id
        self.parent_distance = distance
        self.parent_path = path


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
        return bool(self.children)

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
        self.road_data = road_data

        # Create source node instance
        self.nodes = [Source(source_location[0], source_location[1])]

        # Create other nodes instances
        for idx in range(len(node_locations)):
            loc = node_locations[idx]
            node_id = node_ids[idx] if idx < len(node_ids) else None
            power_demand = node_power_demand[idx]

            self.nodes.append(Nodes(loc, power_demand, node_id))

        # Set network parameters
        self.Vnet = network_voltage
        self.pole_spacing = pole_spacing
        self.Vdrop_max = 13
        self.res_meter = res_per_km / 1000
        self.Imax = max_current

        # Print network initialization details
        # print("Network Designer initialized.")
        # print(f"Source: {self.nodes[0].loc}")
        # print(f"Number of Nodes: {len(self.nodes)}")
        # print(f"Voltage: {self.Vnet}, Pole Spacing: {self.pole_spacing}, "
        #        f"Max Voltage Drop: {self.Vdrop_max}, Resistance/m: {self.res_meter}, Max Current: {self.Imax}")
    @classmethod
    def import_from_csv(cls, cluster_data, road_data, source_coord, network_voltage, pole_spacing,
                        res_per_km, max_current, scl=1, max_V_drop=13):
        """
        Creates a NetworkDesigner instance from CSV data.

        Parameters
        ----------
        cluster_data : list
            List of dictionaries containing cluster data(customers and customer poles).
        road_data : list
            Road data for finding paths.(road nodes and edges)
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
            Maximum allowable voltage drop. Default is 13.
        """
        node_locs = []
        power_demands = []
        node_ids = []

        # Parse cluster data
        for cluster in cluster_data:
            if cluster['Type'] == 'customer_pole':
                node_locs.append((cluster['x'], cluster['y']))
                power_demands.append(cluster['load'])
                node_ids.append(cluster['Cluster'])

        # # Print parsed data
        # print("Parsed CSV Data:")
        # print(f"Node Locations: {node_locs}")
        # print(f"Power Demands: {power_demands}")
        # print(f"Node IDs: {node_ids}")

        return cls(road_data, source_coord, node_locs, power_demands, network_voltage, pole_spacing,
                   res_per_km, max_current, max_V_drop, scl, node_ids)

    def build_network(self):
        """
        Builds network using Esau-Williams CMST.

        """

        self._setup()

        self._cmst()

        self._disconnect_failed()
    def _setup(self):
        """
        Initialization phase for CMST.
        """
        # Create and populate matrices
        self._init_matrices()

        # All nodes are part of their own subtree initially
        self._init_subtrees()

        # Set initial connections from source to nodes
        for i, node in enumerate(self.nodes):
            if isinstance(node, Nodes):
                # Initial parent is always the source
                source_node = self.nodes[0]  # Assuming source is at index 0
                distance = self.distances[0, i]
                path = self.paths[0, i]
                # Set the parent connection, distance, and path
                node.set_parent(source_node, distance, path)
                print(i,node.parent_path,node.parent_distance,'/n')

        # Calculate resistance and test voltage constraints
        self._init_constraints()

        # Update the connection matrix
        self.update_matrix()

        print("\nSETUP DONE!")

    def _init_matrices(self):
        """
        Create connection/distance/resistance/checked paths matrices.
        Uses numpy arrays.
        """
        # Square matrices of size nodes array
        size = (len(self.nodes), len(self.nodes))

        # Create DISTANCE MATRIX
        self.distances = np.zeros(size)
        # Create WEIGHT MATRIX
        self.weights = np.zeros(size)
        # Create CONNECTION MATRIX initially direct source to customers
        self.connections = np.zeros(size, dtype=list)
        # Create PATH CHECKED MATRIX
        self.path_checked = np.zeros(size, dtype=bool)
        # Create ROAD PATH MATRIX
        self.paths = np.zeros(size, dtype=list)

        # Populate distance and weight matrix
        for i, node1 in enumerate(self.nodes):

            for j, node2 in enumerate(self.nodes):
                #connections for Node to itself is set as path checked true
                if i == j:
                    self.path_checked[i, j] = True
                #connections from source to node
                if isinstance(node1,Source) and isinstance(node2,Nodes):
                    direct_distance =  distance_calc(node1.loc[0], node1.loc[1], node2.loc[0], node2.loc[1])
                    weight= 1000 * direct_distance
                    _,path, road_distance = s_path.find_shortest_road_path(self.road_data, node1.loc, node2.loc)


                    if weight < road_distance:
                        self.distances[i,j] = direct_distance
                        self.connections[i,j]= (node1.loc, node2.loc)
                        self.weights[i,j] = weight
                        self.paths[i,j] = (node1.loc, node2.loc)
                    else:
                        self.connections[i,j]= path
                        self.distances[i,j] = road_distance
                        self.weights[i,j] = road_distance
                        self.paths[i,j] = path

    def _init_subtrees(self):
        """
        Sets the subtree of each node to iteself. Only used in initialisation
        phase for Esau-Williams algorithm.

        """
        for idx, node in enumerate(self.nodes):
            if isinstance(node, Source):
                continue
            else:
                node.subtree = idx

    def calculate_res(self, node):
        """
        Calculates the resistance between a node and its parent.

        Parameters
        ----------
        node : Node
            Node object for which line resistance of upstream connection is
            calculated.

        """

        if type(node) == Nodes:
            node_idx = self.nodes.index(node)  # node's index
            parent_idx = node.parent  # node's parent's index
            print(parent_idx, node_idx, )

            # line resistance = res/m * distance
            node.line_res = (self.res_meter * node.parent_distance)

        # if source passed
        else:
            pass

    def _init_constraints(self):
        """
        Initial constraints test before Esau-Williams algorithm is applied.
        Tests if voltage drops across connections are acceptable.

        """

        for node_idx, node in enumerate(self.nodes):

            if isinstance(node, Nodes):

                # calculate resistance between each node and parent (SRC)
                self.calculate_res(node)

                # calculate current drawn by node    I(t) = Pdem(t) / Vnet [DC]
                node.I = node.Pdem / self.Vnet

                # voltage drops is current * line resistance
                voltage_drops = node.line_res * node.I

                if np.max(voltage_drops) > self.Vdrop_max:
                    node.csrt_sat = False
                else:
                    node.csrt_sat = True

                # update path between node and parent (SRC) as checked
                self.path_checked[node_idx, node.parent] = True
                self.path_checked[node.parent, node_idx] = True

            else:
                pass

    def update_matrix(self):
        """
        Updates the distance, weight, and road path matrices using the current path to each subtree's root node.
        """
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):

                if i == 0 or j==0 or i == j:  # ignore source and connections to itself
                    continue  # Use 'continue' instead of 'pass' for clarity
                direct_distance1 = distance_calc(node1.loc[0], node1.loc[1], node2.loc[0], node2.loc[1])
                weight1 = 5 * direct_distance1



                # Initialize variables for tracking the best path and weights
                min_weight = float('inf')
                min_distance = 0
                best_path = []

                # Get the current path from node1 to its parent
                current_path = node1.parent_path

                # print('current path', current_path)

                # Direct connection from node to existing node path
                closest_node, direct_distance2 = s_path.find_closest_node_on_path(current_path, node2.loc)
                weight2 = direct_distance2 * 5 # Assuming weight calculation involves direct distance

                if weight1 < min_weight:
                    min_weight = weight1
                    min_distance = direct_distance2
                    best_path = [node1.loc, node2.loc]

                if weight2 < min_weight:
                    min_weight = weight2
                    min_distance = direct_distance2
                    best_path = [closest_node, node2.loc]

                # Compare with the road connection to the node2
                new_path, road_distance = s_path.find_shortest_path_to_third_node(self.road_data, current_path,
                                                                                  node2.loc)

                # If the road distance is better than the direct connection, update the best path
                if road_distance < min_weight:
                    min_weight = road_distance
                    min_distance = road_distance
                    best_path = new_path

                # Update the matrices with the best found (either direct or road) path
                self.distances[i, j] = min_distance
                self.weights[i, j] = min_weight
                # Save the best path for the current node in the road_path matrix
                self.paths[i, j] = best_path
        # print(self.weights)
        # print(self.paths)
        # print(len(self.weights))

    def _cmst(self):
        """
        Runs CMST algorithm to create network.

        """

        further_improvements = True
        self.old_best_gate = None
        self.old_best_node = None

        loop = 0

        while further_improvements == True:  # and loop < 4:

            loop += 1
            print("-------------------------------")
            print("\nloop " + str(loop))

            print("\nlooking for candidates")

            # find candidate pair
            best_gate_idx, best_node_idx = self._candidate_nodes()

            if best_gate_idx == False and best_node_idx == False:
                print("\nNEW CONNECTION NOT FOUND")
                break

            print("\nsaving state")

            # save current state before making connection
            self._save_state()

            print("\nconnecting nodes")
            print("ATTEMPTING")
            print("gate: " + str(best_gate_idx))
            print("node: " + str(best_node_idx))

            # connect pair
            self._connect_nodes(best_gate_idx, best_node_idx)
            #self.update_matrix()


            print("\ntesting constraints")

            # test constraints on new connection
            # if constraint broken
            if self._test_constraints(best_gate_idx) == False:
                print("\nfailed constraints check, resetting connection")
                # reset the connection
                self._load_prev_state()

    def _candidate_nodes(self):
        """
        Finds two candidate nodes for new connection. Candidate nodes
        are (1) best node to connect to, (2) best gate to connect.

        Returns
        -------
        Node
            Best gate to connect.
        Node
            Best node to connect to.

        """

        best_tradeoff = 0
        best_gate_idx = None
        best_node_idx = None

        for gate_idx, gate in enumerate(self.nodes):

            if type(gate) == Source:
                continue

            if gate.isgate() == False:
                continue

            # gate_idx = self.nodes.index(gate)
            else:
                min_weight = self.weights[0,gate_idx]  # distance gate-SRC

            for node_idx, node in enumerate(self.nodes):

                if type(node) == Source:
                    continue

                if (self.path_checked[node_idx, gate_idx] == True
                        or self.path_checked[gate_idx, node_idx] == True
                        or node.subtree == gate.subtree
                        or node_idx in gate.children):

                    continue

                elif self.weights[gate_idx, node_idx] == 0:
                    continue

                elif self.weights[gate_idx, node_idx] < min_weight:

                    if self.connections[gate_idx, node_idx] == 0:
                        min_weight = self.distances[gate_idx, node_idx]
                        temp_best_node_idx = node_idx
                        temp_best_gate_idx = gate_idx

            tradeoff = self.distances[0,gate_idx] - min_weight

            if tradeoff > 0 and tradeoff > best_tradeoff:
                best_tradeoff = tradeoff
                best_gate_idx = temp_best_gate_idx
                best_node_idx = temp_best_node_idx

        if best_gate_idx == None or best_node_idx == None:  # no new candidates
            return False, False

        else:
            return best_gate_idx, best_node_idx  # new candidates found
    def _save_state(self):
        """
        Saves the network's current state.

        """

        self.prev_nodes = copy.deepcopy(self.nodes)
        self.prev_connections = copy.deepcopy(self.connections)

    def _load_prev_state(self):
        """
        Restores nodes and connections from last saved state.

        """

        self.nodes = self.prev_nodes
        self.connections = self.prev_connections

    def _connect_nodes(self, gate_idx, node_idx):
        """
        Connects a gate (node directly connected to source) with a specified
        node.

        Parameters
        ----------
        gate_idx : int
            Index of joining gate (in nodes array).
        node_idx : int
            Index of joining node (in nodes array).

        """
        # get gate & node objects
        gate = self.nodes[gate_idx]
        node = self.nodes[node_idx]

        # mark path as checked
        self.path_checked[gate_idx, node_idx] = True
        self.path_checked[node_idx, gate_idx] = True

        # mark connection in adjacency matrix
        path = self.paths[node_idx, gate_idx]
        self.connections[node_idx, gate_idx] = path

        print('connection node to source', self.connections[0, node_idx],'\n')
        print('connection gate to source',self.connections[0,gate_idx],'\n')

        # disconnect gate from source in adj matrix
        self.connections[gate_idx, 0] = 0
        self.connections[0, gate_idx] = 0
        print('connection gate to source', self.connections[0, gate_idx],'\n')

        # update subtree for all nodes in gate subtree
        for subnode in self.nodes:
            if type(subnode) == Source:
                continue
            elif subnode == node or subnode == gate:
                continue
            elif subnode.subtree == gate.subtree:

                print("updated subtree of " + str(subnode.node_id))
                print("old subtree: " + str(subnode.subtree))
                print("new subtree: " + str(node.subtree))

                subnode.subtree = node.subtree
            else:
                continue

        # update gate's subtree and parent
        print('gate parent path before', gate.parent)
        gate.parent = node_idx
        gate.subtree = node.subtree
        print('gate parent path after',gate.parent)
        gate.parent_path = path
        gate.parent_distance= self.distances[node_idx,gate_idx]

        node.children.append(gate_idx)  # mark gate as child of node

        # calculate line resistance of new connection
        # note: function calculates resistance of upstream connection, so
        #       passing in gate as argument because it is now downstream.
        self.calculate_res(gate)
        print('connection node to source', self.connections[0, node_idx],'\n')

    def _reset_checks(self):
        """
        Resets check variables of each node in network.

        """

        for node in self.nodes:
            if isinstance(node,Nodes):
                node.I_checked = False
                node.V_checked = False

                node.I_line = np.zeros(len(node.Pdem))

    def _test_current(self, gate_idx):
        """
        Tests currents of all nodes in subtree, starting from gate
        node that has just been connected.

        Parameters
        ----------
        gate_idx : int
            Index of gate node that has been connected to new subtree.

        Returns
        -------
        bool
            True if all currents valid. False if any current invalid.

        """

        active_idx = gate_idx
        active_node = self.nodes[gate_idx]

        constraint_broken = False

        while type(active_node) != Source and constraint_broken == False:

            print(active_idx)

            all_checked = False

            # if active node has children
            if active_node.has_children() == True:

                # search for child with unchecked current
                for child_idx in active_node.children:
                    child = self.nodes[child_idx]

                    # child with unchecked current found
                    # becomes active and stops searching
                    if child.I_checked == False:
                        active_idx = child_idx
                        active_node = child
                        break

                    # all children checked
                    elif child_idx == active_node.children[-1]:
                        all_checked = True

                    else:
                        continue

            # if active node childless or all children have checked currents
            # we are at bottom of subtree
            if active_node.has_children() == False or all_checked == True:

                # current in line = current in child line + current node draws
                if active_node.has_children():

                    I_line_children = 0
                    for child_idx in active_node.children:
                        I_line_children += self.nodes[child_idx].I_line

                    active_node.I_line += active_node.I + I_line_children

                else:
                    active_node.I_line += active_node.I

                # check if current in line above maximum allowable
                if (np.max(active_node.I_line) > self.Imax):
                    constraint_broken = True

                    print("failed current check")

                # mark node as checked
                active_node.I_checked = True

                # move upstream --> parent node becomes active node
                active_idx = active_node.parent
                active_node = self.nodes[active_idx]

        if constraint_broken:
            return False
        else:
            return True

    def _test_voltage(self, gate_idx):
        """
        Tests voltages of all nodes in subtree in which new connection
        has been made.

        Parameters
        ----------
        gate_idx : int
            Index of gate node that has been connected to new subtree.

        Returns
        -------
        bool
            True if all voltages valid. False if any voltage invalid.

        """

        active_idx = self.nodes[gate_idx].subtree
        active_node = self.nodes[active_idx]

        constraint_broken = False

        while type(active_node) != Source and constraint_broken == False:


            # if voltage not checked then calculate voltage
            if active_node.V_checked == False:
                # if active node is gate of subtree
                if active_node.isgate() == True:
                    active_node.V = (self.Vnet
                                     - active_node.I_line
                                     * active_node.line_res)

                # if active node not gate of subtree
                else:
                    parent_node = self.nodes[active_node.parent]
                    active_node.V = (parent_node.V
                                     - active_node.I_line
                                     * active_node.line_res)

                active_node.V_checked = True

                # check constraint
                if np.min(active_node.V) < (self.Vnet - self.Vdrop_max):
                    constraint_broken = True

                    print("voltage check failed")

            elif active_node.V_checked == True:

                if active_node.has_children():

                    for num, child_idx in enumerate(active_node.children):
                        child = self.nodes[child_idx]

                        # child with unchecked voltage found, so stop searching
                        if child.V_checked == False:
                            active_idx = child_idx
                            active_node = child
                            break

                        # all children have checked voltages, move upstream
                        elif (num + 1) == len(active_node.children):
                            active_idx = active_node.parent
                            active_node = self.nodes[active_idx]

                # active node is chidless, move upstream
                elif active_node.has_children() == False:
                    active_idx = active_node.parent
                    active_node = self.nodes[active_idx]

        if constraint_broken:
            return False
        else:
            return True

    def _test_constraints(self, gate_idx):
        return True
        # """
        # Tests constraints on newly connected subtree.
        # Updates constraint attribute in nodes.
        #
        # Parameters
        # ----------
        # gate_idx : int
        #     Index of gate node that has been connected to new subtree..
        #
        # Returns
        # -------
        # bool
        #     True if constraints satisfied. False if constraints broken.
        #
        # """
        #
        # self._reset_checks()
        #
        # print("testing current")
        # I_test = self._test_current(gate_idx)
        #
        # print("testing voltage")
        # V_test = self._test_voltage(gate_idx)
        #
        # gate_node = self.nodes[gate_idx]
        #
        # if I_test == False or V_test == False:
        #     gate_node.csrt_sat = False
        #     return False
        # else:
        #     gate_node.csrt_sat = True
        #     return True

    def _disconnect_failed(self):
        """
        Undoes invalid connections if constraints not satisfied.

        """

        self.final_connect = self.connections.copy()

        for node_idx, node in enumerate(self.nodes):
            if type(node) != Source and node.csrt_sat == False:
                self.final_connect[node_idx, :] = 0
                self.final_connect[:, node_idx] = 0

    def plot_network(self):
        """
        Saves the edges, their distances, and the complete path between connected nodes to a JSON file.
        Each edge contains the latitudes and longitudes of two consecutive nodes in the path.
        """
        edges = []


        # Iterate through all nodes in the network
        for node_idx, node in enumerate(self.nodes):
            if node_idx >0:
                print('node',node_idx)
                print('parent path',node.parent_path)
                print('connection', node.parent)
                path = node.parent_path
                # Ensure the path is valid (not infinity or an invalid marker)

                for i in range(len(path) - 1):
                    node_1 = path[i]  # Current node in the path
                    node_2 = path[i + 1]  # Next node in the path

                    # Append edge details with latitudes and longitudes of two consecutive nodes
                    edges.append({
                        'lat': [float(node_1[1]), float(node_2[1])],  # Latitude (Y-coordinate)
                        'lon': [float(node_1[0]), float(node_2[0])]  # Longitude (X-coordinate)
                    })

        return edges