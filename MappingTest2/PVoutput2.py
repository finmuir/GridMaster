"""

    PV Ouput Subsystem for "Energy 4 Development" VIP

    Code by Alfredo Scalera (alfredo.scalera.2019@uni.strath.ac.uk)

    Based on Python example in renewables.ninja API documentation.
    Original author: Stefan Pfenninger
    Available at: https://www.renewables.ninja/documentation/api/python-example

"""

import requests
import json
import pandas as pd

class PVOutput:
    def __init__(self, lat, long, capacity, year=2019, auto_dataset=True, dataset="merra2", system_loss=0,
                 auto_tilt=True, tilt=40, azim=180):
        self.lat = lat
        self.long = long
        self.capacity = capacity
        self.year = year
        self.auto_dataset = auto_dataset
        self.dataset = dataset
        self.system_loss = system_loss
        self.auto_tilt = auto_tilt
        self.tilt = tilt
        self.azim = azim

    def pv_output(self):
        """
         Parameters
         ----------
         lati : flaot
             Location latitude (deg).
         long : float
             Location longitude (deg).
         year : int
             Data year.
         capacity : flaot
             Capacity of PV panel (Watts).
         dataset : string, optional
             Solar resources dataset.
             Options:
             NASA MERRA-2 ("merra2"):
                 global coverage, 1980 - ongoing.
             CM-SAF SARAH ("sarah"):
                 covers Europe, higher accuracy, 2000 - 2015.
             The default is "merra2".
         system_loss : float, optional
             System's internal losses. Value between 0 and 1.
             The default is 0.
             (future work will account for this)
         auto_tilt : Boolean, optional
             If True, tilt is automatically calculated based on latitude.
             The default is True.
         tilt : flaot, optional
             PV panel tilt angle (deg). The default is 0.
         azim : float, optional
             PV panel azim angle (deg). The default is 180 (facing equator).

         Returns
         -------
         list
             Hourly power output for single PV panel with given capacity in
             selected year.
             """
        if self.auto_dataset:
            self.dataset = self.automatic_dataset()
            print("Dataset: " + self.dataset)

        if self.auto_tilt:
            self.tilt = self.automatic_tilt()
            print("Tilt: " + str(self.tilt))

        start_date = str(self.year) + "-01-01"
        end_date = str(self.year) + "-12-31"

        token = 'fc5b9e4dc8ef24a5923256436575c37dc8ce9195'  # Alfredo's account token.
        url = 'https://www.renewables.ninja/api/data/pv'

        s = requests.session()
        s.headers = {'Authorization': 'Token ' + token}

        args = {
            'lat': self.lat,
            'lon': self.long,
            'date_from': start_date,
            'date_to': end_date,
            'dataset': self.dataset,
            'capacity': self.capacity / 1000,  # from W to kW
            'system_loss': self.system_loss,
            'tracking': 0,
            'tilt': self.tilt,
            'azim': self.azim,
            'header': False,
            'local_time': True,
            'format': 'json'
        }

        print("Retrieving data from Renewables.ninja...")
        r = s.get(url, params=args)
        parsed_response = json.loads(r.text)
        data = pd.read_json(json.dumps(parsed_response), orient='index')

        return [p_out * 1000 for p_out in data["electricity"].values.tolist()]

    def automatic_tilt(self):
        """
            Original author: Stefan Pfenninger
            As found at: https://github.com/renewables-ninja/gsee/blob/master/gsee/pv.py

            Stefan Pfenninger and Iain Staffell (2016).
            Long-term patterns of European PV output using 30 years of
            validated hourly reanalysis and satellite data.
            Energy 114, pp. 1251-1265. doi: 10.1016/j.energy.2016.08.060

            Returns optimal tilt angle for given latitude.
            Works for latitudes between 0 and 50 deg, above 50 deg,
            tilt is set to 40 deg.
            Assumes panel is facing equator (azim = 180 deg)

            Parameters
            ----------
            lati : flaot
                Latitude in degrees.

            Returns
            -------
            float or int
                Optimal tilt angle for equator facing panel in degrees.

            """
        lati = abs(self.lat)

        if lati <= 25:
            return lati * 0.87
        elif lati <= 50:
            return lati * 0.76 + 3.1
        else:
            return 40

    def automatic_dataset(self):
        if (self.lat <= 65 and self.lat >= 24) and (self.long <= 44 and self.long >= -11) and (
                self.year <= 2015 and self.year >= 2000):
            return "sarah"
        else:
            return "merra2"


# Example usage
latitude = -14.24580667  # Latitude of Mthembanji source
longitude = 34.60600833  # Longitude of Mthembanji source
capacity = 5000  # Capacity of PV panel in Watts
year = 2022  # Year of data(most up to date year)
dataset = "merra2"  # Solar resources dataset (optional)

pv_subsystem = PVOutput(latitude, longitude, capacity, year=year, dataset=dataset)
output = pv_subsystem.pv_output()

