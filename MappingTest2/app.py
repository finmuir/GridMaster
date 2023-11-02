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
        a = math.sin(dlat/2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        self.distance = R * c * 1000


points = []


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_parameters():
    return render_template('input.html')


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
    c.execute('CREATE TABLE IF NOT EXISTS points (lat REAL, lng REAL, label TEXT, origin_lat REAL, origin_lng REAL, distance REAL)')
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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    form_data = request.form
    try:
        session['network_voltage'] = network_voltage = float(form_data.get('network_voltage'))
        session['pole_cost'] = pole_cost = float(form_data.get('pole_cost'))
        session['pole_spacing'] = pole_spacing = float(form_data.get('pole_spacing'))
        session['resistance_per_km'] = resistance_per_km = float(form_data.get('resistance_per_km'))
        session['current_rating'] = current_rating = float(form_data.get('current_rating'))
        session['cost_per_km'] = cost_per_km = float(form_data.get('cost_per_km'))
        session['max_voltage_drop'] = max_voltage_drop = float(form_data.get('max_voltage_drop'))
        max_customers = int(form_data.get('max_customers'))



    except TypeError:
        flash("One or more of the input values are missing. Please check your inputs.")
        return redirect(url_for('input_parameters'))
    except ValueError:
        flash("One or more of the input values are invalid. Please check your inputs.")
        return redirect(url_for('input_parameters'))

    # Run the clustering script with the form data
    clusterer = cc.CustomerClustering.import_from_csv(
        "csv_uploads/nodes_datapdem.csv",
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

    # Get the source coordinates
    source_coords = session.get('source_coords', (15, 15))

    if source_coords is not None:
        center_lat = source_coords[0]
        center_lng = source_coords[1]
    else:
        center_lat = 15
        center_lng = 15

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
        'points': map_data,  # Add the map data to the result
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
        max_customers = int(request.form.get('max_customers', 0))

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
        "csv_uploads/nodes_datapdem.csv",
        network_voltage=network_voltage,
        pole_cost=pole_cost,
        pole_spacing=pole_spacing,
        resistance_per_km=resistance_per_km,
        current_rating=current_rating,
        cost_per_km=cost_per_km,
        max_voltage_drop=max_voltage_drop
    )
    clusterer.cluster(max_customers=max_customers)

    net = nd.NetworkDesigner.import_from_csv(
        "csv_uploads/nodes_datapdem.csv",
        network_voltage,
        pole_cost,
        pole_spacing,
        res_per_km,
        max_current,
        cost_per_km,
        max_V_drop=max_volt_drop
    )

    source_coords = (51.505, -0.09)
    net.build_network()
    edges, pos = net.draw_graph()
    code = ''
    for node, coords in pos.items():
        code += f"L.marker([{coords[1]}, {coords[0]}]).addTo(map);\n"
    for edge in edges:
        code += f"L.polyline([[{pos[edge[1]][1]}, {pos[edge[1]][0]}], [{pos[edge[0]][1]}, {pos[edge[0]][0]}]], {{color: 'blue'}}).addTo(map);\n"

    return render_template('networkdesignresult.html', code=code, source_coords=source_coords)


if __name__ == '__main__':
    app.run(debug=True)