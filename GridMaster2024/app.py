from flask import Flask, render_template, jsonify, request, url_for, redirect, flash, session
import sqlite3
import math
from CustomerClusterer import CustomerClustering as cc
#import network_designer as nd
import random
import plotly.io as pio
import plotly.express as px
import pandas as pd
from werkzeug.utils import secure_filename
import os
#from bill_of_quantities import BillOfQuantities
import csv
from flask import Response
#from bill_of_quantities import BillOfQuantities
import io
#from PVoutput import PVOutput
#from Gensizer import GenSizer
import requests
import json
import RoadMapping
from network_dessigner import NetworkDesigner as nd
import networkx as nx


app = Flask(__name__)

app.config['SECRET_KEY'] = 'powergen'

UPLOAD_FOLDER = 'csv_uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#welcome page
@app.route('/')
def welcome():
    return render_template('welcome.html')


#inputs page
@app.route('/clusterinputs', methods=['GET', 'POST'])
def cluster_inputs():
    return render_template('clusterinputs.html')

#cluster results allows user to define desired path and avoid areas
@app.route('/clusteresults', methods=['GET', 'POST'])
def cluster_results():
    # File and form data handling (unchanged)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        session['file_name'] = filename

        # Read the CSV to get the initial coordinates
        df = pd.read_csv(file_path, index_col=0)
        try:
            initial_lat = float(df.loc['X', :].values[0]) if 'X' in df.index else None
            initial_lng = float(df.loc['Y', :].values[0]) if 'Y' in df.index else None
            source_coords = session['source_coords'] = (initial_lng, initial_lat) if initial_lat is not None and initial_lng is not None else (15, 15)
        except (ValueError, IndexError):
            flash("Error reading coordinates from the file.")
            return redirect(url_for('clusterinputs'))
    else:
        source_coords = (15, 15)

    form_data = request.form
    try:
        session['network_voltage'] = network_voltage = float(form_data.get('network_voltage'))
        session['pole_spacing'] = pole_spacing = float(form_data.get('pole_spacing'))
        session['resistance_per_km'] = resistance_per_km = float(form_data.get('resistance_per_km'))
        session['current_rating'] = current_rating = float(form_data.get('current_rating'))
        session['max_voltage_drop'] = max_voltage_drop = float(form_data.get('max_voltage_drop'))
        session['max_customers'] = max_customers = int(form_data.get('max_customers'))
        session['distance_threshold'] = distance_threshold = int(form_data.get('distance_threshold'))
        load_profile_1 = session['load_profile_1'] = json.loads(request.form.get('load_profile_1', '[]'))
        session['load_profile_2'] = json.loads(request.form.get('load_profile_2', '[]'))
        session['load_profile_3'] = json.loads(request.form.get('load_profile_3', '[]'))
        session['load_profile_4'] = json.loads(request.form.get('load_profile_4', '[]'))
        session['load_profile_5'] = json.loads(request.form.get('load_profile_5', '[]'))
        session['load_1_no_customer'] = int(request.form.get('load_1_no_customer', 1))
        session['load_2_no_customer'] = int(request.form.get('load_2_no_customer', 1))
        session['load_3_no_customer'] = int(request.form.get('load_3_no_customer', 1))
        session['load_4_no_customer'] = int(request.form.get('load_4_no_customer', 1))
        session['load_5_no_customer'] = int(request.form.get('load_5_no_customer', 1))

    except TypeError:
        flash("One or more of the input values are missing. Please check your inputs.")
        return redirect(url_for('input_parameters'))
    except ValueError:
        flash("One or more of the input values are invalid. Please check your inputs.")
        return redirect(url_for('input_parameters'))

    distance_threshold = float(request.form.get('distance_threshold', '0'))

    # Run the clustering script with the form data
    clusterer = cc.import_from_csv(
        file_path,
        network_voltage=network_voltage,
        pole_spacing=pole_spacing,
        resistance_per_km=resistance_per_km,
        current_rating=current_rating,
        max_voltage_drop=max_voltage_drop,
        distance_threshold=distance_threshold
    )
    if clusterer is None or not hasattr(clusterer, 'cluster'):
        flash("No customers within the specified distance threshold or an error occurred. Please adjust the threshold or check your input data.")
        return redirect(url_for('cluster_inputs'))

    clusterer.cluster(max_customers=max_customers)

    # Extract distances from each cluster to centroid and prepare cluster data
    cluster_data = []
    for idx, cluster in enumerate(clusterer.clusters):
        x_c = cluster.position[0]
        y_c = cluster.position[1]

        # Collect distances
        distances = cluster.distances
        for i, customer in enumerate(cluster.customers):
            customer_load = int(max(load_profile_1))
            distance_to_centroid = distances[i]

            # Append the customer data with distance to centroid
            cluster_data.append({
                "x": customer.position[0],
                "y": customer.position[1],
                "Cluster": idx,
                "Type": "customer",
                "load": customer_load,
                "Distance_c": distance_to_centroid
            })

        # Append the customer pole data with the total load of the cluster
        total_load = sum(int(max(load_profile_1)) for _ in cluster.customers)
        cluster_data.append({
            "x": x_c,
            "y": y_c,
            "Cluster": idx,
            "Type": "customer_pole",
            "load": total_load
        })



    # Convert cluster data to DataFrame
    df = pd.DataFrame(cluster_data)

    # Road Mapping (unchanged)
    south, west, north, east = RoadMapping.calculate_bounding_box(source_coords[0], source_coords[1], distance_threshold)
    road_result = RoadMapping.find_roads(south, west, north, east)
    nodes_df, edges_df = RoadMapping.format_road_data(road_result)
    # RoadMapping.plot_road_network(nodes_df, edges_df)

    # Add road data to DataFrame
    road_data = []
    for _, row in nodes_df.iterrows():
        road_data.append({"x": row['lon'], "y": row['lat'], "Type": "road_node"})
    for _, row in edges_df.iterrows():
        road_data.append({"x": row['lon'], "y": row['lat'], "Type": "road_edge"})

    road_df = pd.DataFrame(road_data)

    cluster_json = df.to_json(orient='records')
    road_json = road_df.to_json(orient='records')

    return render_template('clusterresults.html', cluster_data=cluster_json, road_data=road_json)





#shows the network

@app.route('/networkdesignerresults', methods=['GET', 'POST'])
def network_design_results():
    # Retrieve session data
    network_voltage = session.get('network_voltage')
    pole_spacing = session.get('pole_spacing')
    resistance_per_km = session.get('resistance_per_km')
    current_rating = session.get('current_rating')
    max_V_drop = session.get('max_voltage_drop')
    max_customers = session.get('max_customers')
    distance_threshold = session.get('distance_threshold')
    load_profile_1 = session.get('load_profile_1')
    load_profile_2 = session.get('load_profile_2')
    load_profile_3 = session.get('load_profile_3')
    load_profile_4 = session.get('load_profile_4')
    load_profile_5 = session.get('load_profile_5')
    source_coords = session.get('source_coords')

    # Load data from form submission
    clusterData = json.loads(request.form.get('clusterData', '[]'))
    roadData = json.loads(request.form.get('roadData', '[]'))


    # Initialize NetworkDesigner with cluster data and other parameters
    net = nd.import_from_csv(
        clusterData,
        roadData,
        source_coords,
        network_voltage,
        pole_spacing,
        resistance_per_km,
        current_rating,
        max_V_drop
    )



        # # Build the final network structure
        # net.build_network()
        # edges, pos = net.draw_graph()

        # Return the results (replace with your actual rendering or JSON response)
    return render_template('network_results.html')



#allows use to input generation sizing parameters
@app.route('/geninputs')
def gensizerinputs():
    return render_template('gensizerinputs.html')

#displays the results of the GenSizer
@app.route('/genoutputs', methods=['POST'])
def gensizer():
    if request.method == 'POST':
        form_data = request.form
        print("Form Data:", form_data)
        try:
            # Get form data with default values
            swarm_size = int(request.form.get('swarm_size', 10))
            panel_capacity = float(request.form.get('panel_capacity', 5000))
            sol_cost = float(request.form.get('sol_cost', 100))
            batt_cost = float(request.form.get('batt_cost', 1000))
            gen_cost = float(request.form.get('gen_cost', 200))
            fuel_cost = float(request.form.get('fuel_cost', 1))
            batt_Wh_max_unit = float(request.form.get('batt_Wh_max_unit', 1000))
            batt_Wh_min_unit = float(request.form.get('batt_Wh_min_unit', 100))
            gen_max_power_out = float(request.form.get('gen_max_power_out', 5000))
            gen_fuel_req = float(request.form.get('gen_fuel_req', 2))
            min_autonomy_days = int(request.form.get('min_autonomy_days', 1))
            max_off_hours = int(request.form.get('max_off_hours', 1))
            pvsystem_loss = float(request.form.get('pvsystem_loss', 1))

            load_profile_1 = session.get('load_profile_1')
            load_profile_2 = session.get('load_profile_2')
            load_profile_3 = session.get('load_profile_3')
            load_profile_4 = session.get('load_profile_4')
            load_profile_5 = session.get('load_profile_5')

            load_1_no_customer = session.get('load_1_no_customer')
            load_2_no_customer = session.get('load_2_no_customer')
            load_3_no_customer = session.get('load_3_no_customer')
            load_4_no_customer = session.get('load_4_no_customer')
            load_5_no_customer = session.get('load_5_no_customer')
            # Append the 24-hour demand array to power_demand 365 times
            # Multiply each value in the day_demand array by number of customers (currently fixed value)
            modified_load_1_demand = [value * load_1_no_customer for value in load_profile_1]
            array_load_1_demand = modified_load_1_demand * 365
            modified_load_2_demand = [value * load_2_no_customer for value in load_profile_2]
            array_load_2_demand = modified_load_2_demand * 365
            modified_load_3_demand = [value * load_3_no_customer for value in load_profile_3]
            array_load_3_demand = modified_load_3_demand * 365
            modified_load_4_demand = [value * load_4_no_customer for value in load_profile_4]
            array_load_4_demand = modified_load_4_demand * 365
            modified_load_5_demand = [value * load_5_no_customer for value in load_profile_5]
            array_load_5_demand = modified_load_5_demand * 365



            power_demand = []
            for i in range(len(array_load_1_demand)):

                power_demand.append(array_load_1_demand[i] + array_load_2_demand[i]+ array_load_3_demand[i]+array_load_4_demand[i] + array_load_5_demand[i])


            print("Yearly Demand Array Length:", len(power_demand))

            print("Yearly Demand Array:", power_demand)

        except KeyError as e:
            return f"Missing form field: {e}"
        except ValueError as e:
            return f"Invalid value for form field: {e}"


        source_coords = session.get('source_coords', (0, 0))
        max_iter = 200
        lat, lon = source_coords
        year = 2023


        #issues with integrating PVOutput


        pv_subsystem = PVOutput(lat, lon, panel_capacity, year=year)
        psol_unit = pv_subsystem.pv_output()
        # Hourly power provided by a single PV panel (come from pv output)
        print(psol_unit)

        # Instantiate GenSizer
        gen_sizer = GenSizer(swarm_size, power_demand, psol_unit,
                             sol_cost, batt_cost, gen_cost, fuel_cost,
                             batt_Wh_max_unit, batt_Wh_min_unit,
                             gen_max_power_out, gen_fuel_req,
                             max_off_hours, min_autonomy_days)
        #run optimise function
        gen_sizer.optimise(max_iter)

        # Get data from GenSizer plot graph function
        solar, batteries, generators, fuel_used, cost, autonomy_days,power_demand,power_battery_discharge,power_battery_charge,power_generator,power_solar ,EbattMin,EbattMax ,dumped_energy, batt_energy= gen_sizer.plot_graphs()

        print('dumped ' , dumped_energy)
        print(len(dumped_energy))

        return render_template('gensizer.html', solar=solar, batteries=batteries, generators=generators,
                               fuel_used=fuel_used, cost=cost, autonomy_days=autonomy_days,
                               power_demand=power_demand,
                               power_battery_discharge =power_battery_discharge,
                               power_battery_charge=power_battery_charge,
                               power_generator=power_generator,
                               power_solar=power_solar,
                               EbattMin=EbattMin,
                               EbattMax=EbattMax ,
                               dumped_energy=dumped_energy,
                               batt_energy=batt_energy)


@app.route('/billofquantities', methods=['GET', 'POST'])
def billofquantities():
    # Render the BOQ template with the calculated values
    return render_template('bill_of_quantities.html', bill_of_quantities=bill_of_quantities, total_cost=total_cost,
                           customer_cost=customer_cost)

