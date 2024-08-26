# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import math

class Customer:

    def __init__(self, customer_id, position, power_demand):
        """
        Customer object for clustering algorithm.

        Parameters
        ----------
        customer_id : string, int or float
            ID for customer.
        position : array_like
            X and Y coordinates of customer in 2D. Shape 2x1.
        power_demand : array_like
            Hourly power demand of customer. 1D array.

        """

        self.customer_id = customer_id
        self.position = tuple(position)
        self.Pdem = np.array(power_demand)

class Cluster:

    def __init__(self, position, customers):
        """
        Cluster object which contains Customer objects.

        Parameters
        ----------
        position : array_like
            X and Y coordinates of cluster centroid in 2D. Shape 2x1.
        customers : array_like
            Array of Customer objects.

        """

        self.position = tuple(position)
        self.customers = list(customers)
        self.n_customers = len(customers)
        self.distances = self._dist_matrix()  # calculate distance matrix

        self.Pdem_total = 0
        for customer in self.customers:
            self.Pdem_total += customer.Pdem

        self.valid = False

    def _dist_matrix(self):
        """
        Creates array populated with distances between customers and centroid
        of cluster. Array is 1D, shape 1 x len(customers).

        Returns
        -------
        Numpy array
            Array populated with internal centroid-customer distances.

        """

        # x and y coordinates of all customers
        X = np.array([customer.position[0] for customer in self.customers])
        Y = np.array([customer.position[1] for customer in self.customers])

        # x and y of centroid
        X_c = self.position[0]
        Y_c = self.position[1]

        # Radius of the Earth in meters
        R = 6371000

        # Converts latitude and longitude points  from degrees to radians
        lat1 = np.radians(Y)
        lon1 = np.radians(X)
        lat2 = np.radians(Y_c)
        lon2 = np.radians(X_c)

        # Haversine formula calculates distance between two point on sphere
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance

    def test_distances(self, max_distance):
        """
        Checks if distances between centroid and customers is valid.
        Affects validity of cluster.

        Parameters
        ----------
        max_distance : float
            Maximum distance allowed between centroid and customers.

        Returns
        -------
        None.

        """

        if np.max(self.distances) > max_distance:
            self.valid = False

        else:
            self.valid = True



    def test_voltages(self, network_voltage, max_voltage_drop, res_per_meter):
        """
        Tests if voltage drops valid in lines connecting centroid and
        customers. Affects validity of cluster.

        Parameters
        ----------
        network_voltage : float
            Voltage at which network operates.
        max_voltage_drop : TYPE
            Maximum allowable voltage drop within clusters.
        res_per_meter : TYPE
            Cable's resistance per meter (ohm/m).

        Returns
        -------
        None.

        """

        for idx, customer in enumerate(self.customers):

            Vdrops = ((customer.Pdem / network_voltage) * res_per_meter
                      * self.distances[idx])

            if np.max(Vdrops) > max_voltage_drop:
                # self.voltage_valid = False
                self.valid = False

                break
            else:
                pass

    def test_max_connections(self, max_connections):
        """
        Checks if number of connections below maximum allowed. Affects
        validity of cluster.

        Parameters
        ----------
        max_connections : int
            Maximum connections (or customers) allowed per cluster.

        Returns
        -------
        None.

        """

        if len(self.customers) > max_connections:

            self.valid = False
        else:
            pass


class InitCluster(Cluster):

    def __init__(self, customers):
        """
        Special cluster object used for first cluster created.
        Centroid is automatically calculated at creation, instead of
        being set as a parameter.

        Parameters
        ----------
        customers : array-like
            Array of Customer objects.

        Returns
        -------
        None.

        """

        self.customers = list(customers)
        self.n_customers = len(customers)
        self._find_centroid()
        self.distances = self._dist_matrix()  # inherited from Cluster
        self.valid = False

    def _find_centroid(self):
        """
        Find centroid of cluster.
        """

        # x and y coordinates of all customers
        X = [customer.position[0] for customer in self.customers]
        Y = [customer.position[1] for customer in self.customers]

        # x and y coordinates of centroid
        self.position = (sum(X) / len(self.customers),
                         sum(Y) / len(self.customers))
class CustomerClustering:

    def __init__(self, init_cluster, source_coord, network_voltage
                 , pole_spacing, resistance_per_km, current_rating,
                  max_voltage_drop=None, max_distance=None):
        """
        Clusters customers together in preparation for network design.

        Parameters
        ----------
        init_cluster : InitCluster
            InitCluster object which initially pools all customers together.
        network_voltage : float
            Voltage at which network operates.
        pole_cost : float
            Cost of electrical pole which will be placed at centroid
            location of cluster and to support line.
        pole_spacing : float
            Space between each electrical pole in meters.
        resistance_per_km : float
            Resistance per kilometer of cable used in ohm/km.
        current_rating : float
            Cable's max current rating.
        cost_per_km : float
            Cable's cost per kilometer.
        max_voltage_drop : float, optional
            Maximum voltage drop allowed between pole and customer.
            If None then maximum voltage drop is dictated by voltage
            regulation.
            The default is None.
        max_distance : float, optional
            Maximum distance allowed between pole and customer
            in meters.
            The default is None.

        """

        # network parameters
        self.network_voltage = network_voltage
        self.source_coord = source_coord
        if max_voltage_drop == None:
            # if none specified, take as 6% of network voltage
            self.max_voltage_drop = 0.06 * network_voltage
        else:
            self.max_voltage_drop = max_voltage_drop

        # pole parameters
        self.max_distance = max_distance
        # self.max_connections = max_connections
        self.pole_spacing = pole_spacing

        # cable parameters
        self.res_m = resistance_per_km / 1000
        self.current_rating = current_rating

        # initialise clusters array
        self.clusters = [init_cluster]

        self.all_clusters_valid = False

    @classmethod
    def import_from_csv(cls, filename, network_voltage
                        , pole_spacing, resistance_per_km,
                        current_rating, scale_factor=1,
                        max_voltage_drop=None, max_distance=None, distance_threshold=None):
        """
        Creates CustomerCLustering object and generates Customer
        objects based on data within specified CSV file.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        filename : string
            Name of CSV file containing customer information.
            Must follow default format.
        network_voltage : float
            Voltage at which network operates.
        pole_cost : float
            Cost of electrical pole which will be placed at centroid
            location of cluster and to support line.
        pole_spacing : float
            Space between each electrical pole in meters.
        resistance_per_km : float
            Resistance per kilometer of cable used in ohm/km.
        current_rating : float
            Cable's max current rating.
        cost_per_km : float
            Cable's cost per kilometer.
        max_voltage_drop : float, optional
            Maximum voltage drop allowed between pole and customer.
            If None then maximum voltage drop is dictated by voltage
            regulation.
            The default is None.
        max_distance : float, optional
            Maximum distance allowed between pole and customer
            in meters.
            The default is None.

        Returns
        -------
        CustomerClustering
            CustomerClustering object is returned with given parameters.

        """
        # read csv file as pandas dataframe
        df = pd.read_csv(str(filename))
        df = df.set_index("ID")

        # import customers and create initial single cluster
        customers = []
        source_coord = None
        source = True

        for customer_id, data in df.items():
            if source:
                source_coord = (float(data[0]), float(data[1]))
                source = False
            else:
                position = (scale_factor * float(data["X"]), scale_factor * float(data["Y"]))
                power_demand = data[2:]
                if distance_threshold is not None:
                    customer_distance = cls.calculate_distance(source_coord[0], source_coord[1], position[0], position[1])
                    if customer_distance <= distance_threshold:
                        customers.append(Customer(customer_id, position, power_demand))

                        # Right before initiating InitCluster
        print(f"Number of customers being passed to InitCluster: {len(customers)}")
        if not customers:
            print("No customers within the specified distance threshold.")
            # Handle the case appropriately, e.g., by returning None or raising an exception
            return None  # Or raise an exception
        else:
            init_cluster = InitCluster(customers)

        return cls(init_cluster, source_coord, network_voltage,
                   pole_spacing, resistance_per_km, current_rating
                   ,max_voltage_drop=max_voltage_drop,
                   max_distance=max_distance)

    @staticmethod
    def calculate_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance * 1000


    @classmethod
    def import_from_OTHER(cls):
        # method that will produce customers (with their respective locations and power demands) from Map
        # PLACEHOLDER
        pass

    def cluster(self, max_customers=6):
        """
        Clusters customers together based on proximity and finds
        centroid for each cluster.

        Parameters
        ----------
        max_customers : int, optional
            Maximum number of customers per cluster, default is 6
        """

        while self.all_clusters_valid == False:

            # test constraints on all clusters
            self._test_constraints_all(max_customers)  # updates value of all_clusters_valid

            # keep valid and apply kmeans (k=2) to invalid clusters
            new_clusters = []
            for cluster in self.clusters:
                if cluster.valid == True:  # keep valid clusters
                    new_clusters.append(cluster)
                elif cluster.valid == False:
                    # apply kmean to invalid cluster and add new ones
                    new_clusters += self._apply_kmeans(cluster)

            self.clusters = new_clusters

        # !!!
        self._merge_loop(max_customers)


    # def _total_cost(self):
    #     """
    #     Calculates the total cost of the clustering setup.
    #
    #     """
    #
    #     # array of all distances
    #     d = np.array([cluster.distances for cluster in self.clusters],
    #                  dtype=object)
    #     # concatenating all arrays and summing all elements
    #     self.total_cable_length = float(np.sum(np.concatenate(d)))
    #
    #     line_cost = self.total_cable_length * self.cost_m
    #     num_poles = math.ceil(self.total_cable_length / self.pole_spacing)
    #     num_poles += len(self.clusters)
    #     poles_cost = num_poles * self.pole_cost
    #
    #     self.total_cost = line_cost + poles_cost

    def _test_constraints_all(self, max_customers):
        """
        Tests constraints on all clusters.

        """

        self.all_clusters_valid = True  # assume all clusters valid initially

        for cluster in self.clusters:

            cluster.valid = True  # assume cluster valid initially

            # test constraints - these methods update cluster.valid
            if self.max_distance != None:  # if max distance specified
                cluster.test_distances()
            cluster.test_voltages(self.network_voltage, self.max_voltage_drop,
                                  self.res_m)
            # cluster.test_max_connections(self.max_connections)
            cluster.test_max_connections(max_customers)

            if cluster.valid == False:
                self.all_clusters_valid = False

    def _apply_kmeans(self, cluster):
        """
        Splits cluster into two new clusters by applying kmeans with
        k = 2 (two clusters).

        Parameters
        ----------
        cluster : Cluster
            Cluster object to be split.

        Returns
        -------
        new_clusters : list
            List containing two new cluster objects.

        """

        pos = np.array([customer.position for customer in cluster.customers])

        # apply kmeans to invalid clusters (k = 2)
        kmeans = KMeans(n_clusters=2).fit(pos)

        cluster_centers = kmeans.cluster_centers_
        cust_labels = kmeans.labels_
        new_clusters = []

        for ce_label, center in enumerate(cluster_centers):
            customers = []
            for cu_idx, customer in enumerate(cluster.customers):
                # if customer label = centroid label
                if cust_labels[cu_idx] == ce_label:
                    customers.append(customer)

            # create new cluster
            new_clusters.append(Cluster(center, customers))

        return new_clusters

    def _merge_loop(self, max_customers):
        """
        Attempts merging clusters together in order to reduce number
        of clusters.

        """

        print("\nAttempting merge")

        self._dist_matrix = self._init_dist_matrix(max_customers)

        further_imp = True
        while further_imp:

            # find indices of closest pair
            idx_1, idx_2 = np.unravel_index(self._dist_matrix.argmin(),
                                            self._dist_matrix.shape)

            cluster_1 = self.clusters[idx_1]
            cluster_2 = self.clusters[idx_2]
            customers = cluster_1.customers + cluster_2.customers

            new_cluster = InitCluster(customers)
            self._test_constraints(new_cluster, max_customers)

            if new_cluster.valid == True:
                # remove old clusters and add new one
                self.clusters.remove(cluster_1)
                self.clusters.remove(cluster_2)
                self.clusters.append(new_cluster)

                # create new distance matrix
                self._dist_matrix = self._init_dist_matrix(max_customers)

            elif new_cluster.valid == False:
                self._dist_matrix[idx_1, idx_2] = np.inf
                self._dist_matrix[idx_2, idx_1] = np.inf

            if np.isinf(self._dist_matrix).all():
                further_imp = False

        print("\nFinished merge attempt")

    def _test_constraints(self, cluster, max_customers):
        """
        Tests maximum distance (if specified), maximum voltage and
        maximum customers constraints on cluster.

        Parameters
        ----------
        cluster : Cluster
            Cluster object on which constraints tested.

        """

        for cluster in self.clusters:

            cluster.valid = True  # assume cluster valid initially

            # test constraints - these methods update cluster.valid
            if self.max_distance != None:  # if max distance specified
                cluster.test_distances()
            cluster.test_voltages(self.network_voltage, self.max_voltage_drop,
                                  self.res_m)
            cluster.test_max_connections(max_customers)

    def _init_dist_matrix(self, max_customers):
        """
        Creates distance matrix containing distances between clusters.
        Used for merging process. Pairs to i

        Returns
        -------
        dist_matrix : Numpy array
            Matrix containing distances between clusters.

        """
        # create distance matrix (distances between clusters)
        # used for merging process
        # pairs to ignore marked with inf

        size = (len(self.clusters), len(self.clusters))
        dist_matrix = np.full(size, np.inf)  # all values initially NaN
        for idx_1, cluster_1 in enumerate(self.clusters):



            # skip cluster if it already has maximum number of customers
            if cluster_1.n_customers == max_customers:


                continue

            # position of first cluster
            X_1 = cluster_1.position[0]
            Y_1 = cluster_1.position[1]

            for idx_2, cluster_2 in enumerate(self.clusters):

                if idx_1 == idx_2:

                    continue

                elif ((cluster_1.n_customers + cluster_2.n_customers)
                      > max_customers):

                    continue

                # position of second cluster
                X_2 = cluster_2.position[0]
                Y_2 = cluster_2.position[1]

                # Radius of the Earth in meters
                R = 6371000

                # Converts latitude and longitude points  from degrees to radians
                lat1 = np.radians(Y_1)
                lon1 = np.radians(X_1)
                lat2 = np.radians(Y_2)
                lon2 = np.radians(X_2)

                # Haversine formula calculates distance between two point on sphere
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist = R * c

                if self.max_distance != None and dist > self.max_distance:

                    continue

                else:
                    dist_matrix[idx_1, idx_2] = dist

        return dist_matrix