from flask import Flask, render_template, jsonify, request, url_for, redirect, flash, session
import sqlite3
import math
import customer_clustering as cc
import network_designer as nd
import random
import plotly.io as pio
import plotly.express as px
import pandas as pd
from werkzeug.utils import secure_filename
import os
import csv
from flask import Response
from GridMaster2024.bill_of_quantities import BillOfQuantities
import io
from PVoutput2 import PVOutput
from Gensizer2 import GenSizer
import json



app = Flask(__name__)

app.config['SECRET_KEY'] = 'powergen'

UPLOAD_FOLDER = 'csv_uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class Point:
    def __init__(self, lat, lng, label, origin=None):
        self.lat = lat
        self.lng = lng
        self.label = label
        self.origin = origin
        self.distance = 0

    def set_origin(self, origin):
        self.origin = origin
        self.calculate_distance()

    def calculate_distance(self):
        R = 6371  # Radius of the Earth in kilometers
        lat1, lng1 = math.radians(self.lat), math.radians(self.lng)
        lat2, lng2 = math.radians(
            self.origin.lat), math.radians(self.origin.lng)
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlng / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        self.distance = R * c * 1000


points = []


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/store-source-point', methods=['POST'])
def store_source_point():
    source_lat = float(request.form['source_lat'])
    source_lng = float(request.form['source_lng'])

    session['source_coords'] = (source_lat, source_lng)

    return redirect(url_for('cluster_inputs'))


@app.route('/store-point', methods=['POST'])
def store_point():
    lat = float(request.form['lat'])
    lng = float(request.form['lng'])
    label = request.form['label']

    if len(points) == 0:
        point = Point(lat, lng, label)
        points.append(point)
    else:
        point = Point(lat, lng, label, points[0])
        point.set_origin(points[0])
        points.append(point)

    conn = sqlite3.connect('points.db')
    c = conn.cursor()
    c.execute(
        'CREATE TABLE IF NOT EXISTS points (lat REAL, lng REAL, label TEXT, origin_lat REAL, origin_lng REAL, distance REAL)')
    if point.origin is None:
        c.execute('INSERT INTO points (lat, lng, label, origin_lat, origin_lng, distance) VALUES (?, ?, ?, ?, ?, ?)',
                  (lat, lng, label, None, None, 0))
    else:
        c.execute('INSERT INTO points (lat, lng, label, origin_lat, origin_lng, distance) VALUES (?, ?, ?, ?, ?, ?)',
                  (lat, lng, label, point.origin.lat, point.origin.lng, point.distance))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


@app.route('/clusterinputs', methods=['GET', 'POST'])
def cluster_inputs():
    return render_template('clusterinputs.html')

@app.route('/plot-data', methods=['GET', 'POST'])
def plot_data():
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
            # Extract the first set of coordinates and correct their order
            initial_lat = float(df.loc['X', :].values[0]) if 'X' in df.index else None
            initial_lng = float(df.loc['Y', :].values[0]) if 'Y' in df.index else None
            # Ensure that the order is longitude first, then latitude
            source_coords = (initial_lng, initial_lat) if initial_lat is not None and initial_lng is not None else (
            15, 15)
        except (ValueError, IndexError):
            # Handle cases where conversion to float fails or values are not found
            flash("Error reading coordinates from the file.")
            return redirect(url_for('cluster_inputs'))
    else:
        # Default coordinates if no file is uploaded
        source_coords = (15, 15)

    form_data = request.form
    try:
        session['network_voltage'] = network_voltage = float(form_data.get('network_voltage'))
        session['pole_cost'] = pole_cost = float(form_data.get('pole_cost'))
        session['pole_spacing'] = pole_spacing = float(form_data.get('pole_spacing'))
        session['resistance_per_km'] = resistance_per_km = float(form_data.get('resistance_per_km'))
        session['current_rating'] = current_rating = float(form_data.get('current_rating'))
        session['cost_per_km'] = cost_per_km = float(form_data.get('cost_per_km'))
        session['max_voltage_drop'] = max_voltage_drop = float(form_data.get('max_voltage_drop'))
        session['max_customers'] = max_customers = int(form_data.get('max_customers'))
        session['distance_threshold'] = distance_threshold = int(form_data.get('distance_threshold'))
        #session['labor_cost'] = labor_cost = float(form_data.get('labor_cost'))

    except TypeError:
        flash("One or more of the input values are missing. Please check your inputs.")
        return redirect(url_for('input_parameters'))
    except ValueError:
        flash("One or more of the input values are invalid. Please check your inputs.")
        return redirect(url_for('input_parameters'))

    distance_threshold = float(request.form.get('distance_threshold', '0'))

    # Run the clustering script with the form data
    clusterer = cc.CustomerClustering.import_from_csv(
        file_path,
        network_voltage=network_voltage,
        pole_cost=pole_cost,
        pole_spacing=pole_spacing,
        resistance_per_km=resistance_per_km,
        current_rating=current_rating,
        cost_per_km=cost_per_km,
        max_voltage_drop=max_voltage_drop,
        distance_threshold=distance_threshold
    )

    if clusterer is None or not hasattr(clusterer, 'cluster'):
        flash("No customers within the specified distance threshold or an error occurred. Please adjust the threshold or check your input data.")
        return redirect(url_for('cluster_inputs'))

    clusterer.cluster(max_customers=max_customers)

    # Prepare the data for the plot
    cluster_data = []
    for idx, cluster in enumerate(clusterer.clusters):
        x_c = cluster.position[0]
        y_c = cluster.position[1]
        x = [customer.position[0] for customer in cluster.customers]
        y = [customer.position[1] for customer in cluster.customers]
        color = random.randint(0, 500)
        for i in range(len(x)):
            cluster_data.append(
                {"x": x[i], "y": y[i], "Cluster": idx, "Type": "customer"})
        cluster_data.append(
            {"x": x_c, "y": y_c, "Cluster": idx, "Type": "pole"})

    df = pd.DataFrame(cluster_data)

    # Set up your Mapbox access token
    px.set_mapbox_access_token(
        'pk.eyJ1IjoiZmlubXVpciIsImEiOiJjbGppaGV2amgwMDhzM2RwcGE5eXllanM0In0.JGhdq86XC-ShgG2lokibfw')

    # Create a scatter plot on a Mapbox map
    fig = px.scatter_mapbox(df, lat='y', lon='x', color='Cluster',
                            hover_data=['Type'], zoom=10, height=500)

    # Convert fig to JSON
    fig_json = pio.to_json(fig)

    map_data = []
    for point in cluster_data:
        map_data.append(
            {'lat': point['y'], 'lon': point['x'], 'type': point['Type']})

    # Return the results to the client
    result = {
        'fig_json': fig_json,
        'number_of_clusters': len(clusterer.clusters),
        'total_line_cost': clusterer.total_cost,
        'points': map_data,
    }

    return render_template('clusterresults.html', result=result, source_coords=source_coords)



@app.route('/plot-data-network', methods=['GET', 'POST'])
def plot_data_network():
    try:
        network_voltage = session.get('network_voltage', 0)
        pole_cost = session.get('pole_cost', 0)
        pole_spacing = session.get('pole_spacing', 0)
        resistance_per_km = session.get('resistance_per_km', 0)
        current_rating = session.get('current_rating', 0)
        cost_per_km = session.get('cost_per_km', 0)
        max_voltage_drop = session.get('max_voltage_drop', 0)
        max_customers = session.get('max_customers', 0)
        distance_threshold = session.get('distance_threshold', 0)

        file_name = session.get('file_name', None)
        if not file_name or file_name == 0:
            flash("File name is missing. Please upload the file.")
            return redirect(url_for('cluster_inputs'))

    except TypeError:
        flash("One or more of the input values are missing. Please check your inputs.")
        return redirect(url_for('input_parameters'))
    except ValueError:
        flash("One or more of the input values are invalid. Please check your inputs.")
        return redirect(url_for('input_parameters'))

    network_voltage = network_voltage  # V
    pole_cost = pole_cost  # £
    pole_spacing = pole_spacing  # m
    res_per_km = resistance_per_km  # ohm/km
    max_current = current_rating  # A
    cost_per_km = cost_per_km  # £/km
    max_volt_drop = max_voltage_drop  # V
    distance_threshold = distance_threshold  # m

    clusterer = cc.CustomerClustering.import_from_csv(
        f"csv_uploads/{file_name}",
        network_voltage=network_voltage,
        pole_cost=pole_cost,
        pole_spacing=pole_spacing,
        resistance_per_km=res_per_km,
        current_rating=max_current,
        cost_per_km=cost_per_km,
        max_voltage_drop=max_volt_drop,
        distance_threshold=distance_threshold
    )

    if clusterer is None or not hasattr(clusterer, 'cluster'):
        flash(
            "No customers within the specified distance threshold or an error occurred. Please adjust the threshold or check your input data.")
        return redirect(url_for('cluster_inputs'))

    clusterer.cluster(max_customers=max_customers)

    net = nd.NetworkDesigner.import_from_csv(
        clusterer.clusters,
        clusterer.source_coord,
        network_voltage,
        pole_cost,
        pole_spacing,
        res_per_km,
        max_current,
        cost_per_km,
        max_V_drop=max_volt_drop
    )

    net.build_network()
    edges, pos = net.draw_graph()
    code = ''

    customer_icons = ['mapbox-maki-93d5dd4/icons/markerblue.svg', 'mapbox-maki-93d5dd4/icons/markercharcoal.svg',
                      'mapbox-maki-93d5dd4/icons/markerdarkblue.svg',
                      'mapbox-maki-93d5dd4/icons/markerdarkgreen.svg', 'mapbox-maki-93d5dd4/icons/markerdarkyellow.svg',
                      'mapbox-maki-93d5dd4/icons/markergreen.svg',
                      'mapbox-maki-93d5dd4/icons/markergrey.svg', 'mapbox-maki-93d5dd4/icons/markerlightblue.svg',
                      'mapbox-maki-93d5dd4/icons/markerlightgreen.svg',
                      'mapbox-maki-93d5dd4/icons/markerlightpurple.svg', 'mapbox-maki-93d5dd4/icons/markerlightred.svg',
                      'mapbox-maki-93d5dd4/icons/markerorange.svg',
                      'mapbox-maki-93d5dd4/icons/markerpink.svg', 'mapbox-maki-93d5dd4/icons/markerpurple.svg',
                      'mapbox-maki-93d5dd4/icons/markerred.svg', 'mapbox-maki-93d5dd4/icons/markerwhite.svg',
                      'mapbox-maki-93d5dd4/icons/markeryellow.svg']

    source = True
    for node, coords in pos.items():
        if source:
            markerIcon = "L.icon({iconUrl: 'static/images/mapbox-maki-93d5dd4/icons/charging-station.svg', iconSize: [25, 41], iconAnchor: [12.5,20.5]})"
            code += f"L.marker([{coords[1]}, {coords[0]}], {{icon: {markerIcon}}}).addTo(map);\n"
            source = False
        else:
            markerIcon = "L.icon({iconUrl: 'static/images/mapbox-maki-93d5dd4/icons/observation-tower.svg', iconSize: [22, 31], iconAnchor: [12.5, 20.5]})"
            code += f"L.marker([{coords[1]}, {coords[0]}], {{icon: {markerIcon}}}).addTo(map);\n"

    for edge in edges:
        code += f"L.polyline([[{pos[edge[1]][1]}, {pos[edge[1]][0]}], [{pos[edge[0]][1]}, {pos[edge[0]][0]}]], {{color: '#2c353b'}}).addTo(map);\n"

    # Iterate over each cluster and use different customer icon
    for idx, cluster in enumerate(clusterer.clusters):
        pole_position = cluster.position
        customer_icon_file = customer_icons[idx % len(customer_icons)]
        for customer in cluster.customers:
            customer_position = customer.position
            # Construct the full path for the icon
            customer_icon = f"static/images/{customer_icon_file}"
            code += f"L.marker([{customer_position[1]}, {customer_position[0]}], {{icon: L.icon({{iconUrl: '{customer_icon}', iconSize: [20, 27], iconAnchor: [12.5, 20.5]}})}}).addTo(map);\n"

    for cluster in clusterer.clusters:
        pole_position = cluster.position
        for customer in cluster.customers:
            customer_position = customer.position
            code += f"L.polyline([[{pole_position[1]}, {pole_position[0]}], [{customer_position[1]}, {customer_position[0]}]], {{color: '#4a2900'}}).addTo(map);\n"
    session['number_of_poles'] = request.form.get('number_of_poles', type=int)
    session['t_poles'] = int(request.form.get('t_poles', 0))
    return render_template('networkdesignresult.html', code=code, source_coords=clusterer.source_coord)


@app.route('/geninputs')
def gensizerinputs():


    return render_template('gensizerinputs.html')


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
            load_profile_1 = json.loads(request.form.get('load_profile_1', '[]'))
            load_profile_2 = json.loads(request.form.get('load_profile_2', '[]'))
            load_profile_3 = json.loads(request.form.get('load_profile_3', '[]'))
            load_profile_4 = json.loads(request.form.get('load_profile_4', '[]'))
            load_profile_5 = json.loads(request.form.get('load_profile_5', '[]'))
            power_demand = []
            load_1_no_customer = int(request.form.get('load_1_no_customer', 1))
            load_2_no_customer = int(request.form.get('load_2_no_customer', 1))
            load_3_no_customer = int(request.form.get('load_3_no_customer', 1))
            load_4_no_customer = int(request.form.get('load_4_no_customer', 1))
            load_5_no_customer = int(request.form.get('load_5_no_customer', 1))
            number_of_customers = (session.get('number_of_customers', 1))

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



        #Generate example PV output
        #psol_unit = [0.0, 0.0, 0.0, 7.0, 74.0, 190.0, 345.00000000000006, 498.0, 594.0, 657.0, 611.0, 548.0, 459.0, 298.0, 134.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,] * 365

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
@app.route('/download-boq')
def download_boq():
    data = BillOfQuantities.generate_bill_of_quantities()
    proxy = io.StringIO()

    writer = csv.writer(proxy)
    writer.writerow(['Item', 'Cost'])  # Writing the header
    for item, cost in data.items():
        writer.writerow([item, cost])

    proxy.seek(0)
    return Response(
        proxy.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=bill_of_quantities.csv',
            'Content-type': 'text/csv'
        }
    )
@app.route('/billofquantities', methods=['GET', 'POST'])
def billofquantities():
    if request.method == 'POST':
        # Get T-Poles from the form, defaulting to 0 if not provided
        t_poles = int(request.form.get('t_poles', 0))
        labor_cost = float(request.form.get('labor_cost', 20))  # Default labor cost
        pole_cost = float(request.form.get('pole_cost', 10))  # Get pole cost from form

        number_of_poles = session.get('number_of_poles', 0)  # Number of poles from session
        if number_of_poles <= 0:
            flash("Number of poles not found. Please restart the process.")
            return redirect(url_for('cluster_inputs'))

        # Define quantities and costs including the dynamic calculation for Bobbin Insulators
        quantities_and_costs = {
            "Nine Metre Poles": (number_of_poles, pole_cost, 2),
            "LV Insulators": (number_of_poles, 20, 1.5),
            # Bobbin Insulators calculation now uses t_poles directly
            "Bobbin Insulators": ((2 * number_of_poles) + (2 * t_poles), 5, 0.5),
            "Stay Blocks": (2 * number_of_poles, 8, 1),
            "D Irons": (4 * number_of_poles, 15, 1.5),
            "Four-way Boards": (number_of_poles, 12, 2),
            "50mm ACC Conductor": (6000, 2, 10),#needs distances
            "16mm Twin Figure 8": (1000, 3, 5),#needs distances
            "2x16mm Armored Cable": (80, 5, 4),#needs distances
            "PG Clamp AL/AL": (max(1, number_of_poles // 2), 1, 1.5),
            "PG Clamp AL/CU": (max(1, number_of_poles // 2), 1, 1.5),
            "Earth Rod": (number_of_poles, 4, 2.5),
        }

        # Instantiate the BillOfQuantities with the updated quantities and costs
        boq = BillOfQuantities(quantities_and_costs, labor_cost)

        bill_of_quantities =boq.generate_bill_of_quantities()

        total_cost = sum(item['Total Cost'] for item in bill_of_quantities.values())
        customer_cost = total_cost / (session.get('number_of_customers', 1))




    # Render the BOQ template with the calculated values
    return render_template('bill_of_quantities.html', bill_of_quantities=bill_of_quantities, total_cost=total_cost,
                           customer_cost=customer_cost)




if __name__ == '__main__':
    app.run(debug=True)
