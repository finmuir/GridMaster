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
from bill_of_quantities import BillOfQuantities
import csv
from flask import Response
from bill_of_quantities import generate_bill_of_quantities
import io

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

@app.route('/geninputs')
def gensizer():
    return render_template('gensizerinputs.html')

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

    except TypeError:
        flash("One or more of the input values are missing. Please check your inputs.")
        return redirect(url_for('input_parameters'))
    except ValueError:
        flash("One or more of the input values are invalid. Please check your inputs.")
        return redirect(url_for('input_parameters'))

    # Run the clustering script with the form data
    clusterer = cc.CustomerClustering.import_from_csv(
        file_path,
        network_voltage=network_voltage,
        pole_cost=pole_cost,
        pole_spacing=pole_spacing,
        resistance_per_km=resistance_per_km,
        current_rating=current_rating,
        cost_per_km=cost_per_km,
        max_voltage_drop=max_voltage_drop
    )
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
        file_name = session.get('file_name', 0)

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

    clusterer = cc.CustomerClustering.import_from_csv(
        f"csv_uploads/{file_name}",
        network_voltage=network_voltage,
        pole_cost=pole_cost,
        pole_spacing=pole_spacing,
        resistance_per_km=res_per_km,
        current_rating=max_current,
        cost_per_km=cost_per_km,
        max_voltage_drop=max_volt_drop
    )
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

    customer_icons = ['mapbox-maki-93d5dd4/icons/markerblue.svg', 'mapbox-maki-93d5dd4/icons/markercharcoal.svg', 'mapbox-maki-93d5dd4/icons/markerdarkblue.svg',
                      'mapbox-maki-93d5dd4/icons/markerdarkgreen.svg', 'mapbox-maki-93d5dd4/icons/markerdarkyellow.svg', 'mapbox-maki-93d5dd4/icons/markergreen.svg',
                      'mapbox-maki-93d5dd4/icons/markergrey.svg', 'mapbox-maki-93d5dd4/icons/markerlightblue.svg', 'mapbox-maki-93d5dd4/icons/markerlightgreen.svg',
                      'mapbox-maki-93d5dd4/icons/markerlightpurple.svg', 'mapbox-maki-93d5dd4/icons/markerlightred.svg', 'mapbox-maki-93d5dd4/icons/markerorange.svg',
                      'mapbox-maki-93d5dd4/icons/markerpink.svg', 'mapbox-maki-93d5dd4/icons/markerpurple.svg', 'mapbox-maki-93d5dd4/icons/markerred.svg', 'mapbox-maki-93d5dd4/icons/markerwhite.svg',
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
        customer_icon_file = customer_icons[idx % len(customer_icons)]  # Choose icon file
        for customer in cluster.customers:
            customer_position = customer.position
            # Construct the full path for the icon
            customer_icon = f"static/images/{customer_icon_file}"
            code += f"L.marker([{customer_position[1]}, {customer_position[0]}], {{icon: L.icon({{iconUrl: '{customer_icon}', iconSize: [20, 27], iconAnchor: [12.5, 20.5]}})}}).addTo(map);\n"

    for cluster in clusterer.clusters:
        pole_position = cluster.position  # Centroid of the cluster (longitude, latitude)
        for customer in cluster.customers:
            customer_position = customer.position  # Customer's position (longitude, latitude)
            code += f"L.polyline([[{pole_position[1]}, {pole_position[0]}], [{customer_position[1]}, {customer_position[0]}]], {{color: '#4a2900'}}).addTo(map);\n"

    return render_template('networkdesignresult.html', code=code, source_coords=clusterer.source_coord)

@app.route('/download-boq')
def download_boq():
    data = generate_bill_of_quantities()  # Assuming this function returns your bill of quantities data as a dictionary
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
    # Parameters for BillOfQuantities - you might want to retrieve these from the session or a form
    cable_cost_per_km = 1000
    cable_length_km = 5
    pole_cost = 500
    num_customers = 10
    daily_running_cost = 50
    installation_cost = 5000

    # Create an instance of the BillOfQuantities with the given parameters
    boq = BillOfQuantities(cable_cost_per_km, cable_length_km, pole_cost, num_customers, daily_running_cost, installation_cost)

    # Generate the bill of quantities
    bill_of_quantities = boq.generate_bill_of_quantities()

    # Render the bill_of_quantities.html template, passing in the bill_of_quantities dictionary
    return render_template('bill_of_quantities.html', bill_of_quantities=bill_of_quantities)


if __name__ == '__main__':
    app.run(debug=True)