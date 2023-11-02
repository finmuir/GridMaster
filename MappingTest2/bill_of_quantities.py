class BillOfQuantities:
    def __init__(self, cable_cost_per_km, cable_length_km, pole_cost, num_customers, daily_running_cost, installation_cost):
        self.cable_cost_per_km = cable_cost_per_km
        self.cable_length_km = cable_length_km
        self.pole_cost = pole_cost
        self.num_customers = num_customers
        self.daily_running_cost = daily_running_cost
        self.installation_cost = installation_cost

    def calculate_cable_cost(self):
        cable_cost = self.cable_cost_per_km * self.cable_length_km
        return cable_cost

    def calculate_pole_cost(self):
        pole_cost_total = self.pole_cost * self.num_customers  # Assuming one pole per customer
        return pole_cost_total

    def calculate_total_installation_cost(self):
        total_installation_cost = self.calculate_cable_cost() + self.calculate_pole_cost() + self.installation_cost
        return total_installation_cost

    def calculate_daily_operational_cost(self):
        return self.daily_running_cost

    def generate_bill_of_quantities(self):
        cable_cost = self.calculate_cable_cost()
        pole_cost = self.calculate_pole_cost()
        installation_cost = self.installation_cost
        operational_cost = self.calculate_daily_operational_cost()
        total_cost = cable_cost + pole_cost + installation_cost

        return {
            "Cable Cost (per km)": self.cable_cost_per_km,
            "Cable Length (km)": self.cable_length_km,
            "Cable Cost Total": cable_cost,
            "Pole Cost per Customer": self.pole_cost,
            "Number of Customers": self.num_customers,
            "Pole Cost Total": pole_cost,
            "Installation Cost": installation_cost,
            "Operational Cost (Daily)": operational_cost,
            "Total Installation Cost": total_cost
        }


# Example usage:
cable_cost_per_km = 1000  # Specify your cable cost per km
cable_length_km = 5  # Specify the cable length in km
pole_cost = 500  # Specify pole cost
num_customers = 10  # Specify the number of customers
daily_running_cost = 50  # Specify daily running cost
installation_cost = 5000  # Specify installation cost

boq = BillOfQuantities(cable_cost_per_km, cable_length_km, pole_cost, num_customers, daily_running_cost, installation_cost)
bill_of_quantities = boq.generate_bill_of_quantities()

for item, value in bill_of_quantities.items():
    print(f"{item}: {value}")
def generate_bill_of_quantities():
    cable_cost_per_km = 1000
    cable_length_km = 5
    pole_cost = 500
    num_customers = 10
    daily_running_cost = 50
    installation_cost = 5000

    boq = BillOfQuantities(cable_cost_per_km, cable_length_km, pole_cost, num_customers, daily_running_cost, installation_cost)
    bill_of_quantities = boq.generate_bill_of_quantities()

    return bill_of_quantities
