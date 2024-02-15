from flask import Flask, render_template

app = Flask(__name__)

class BillOfQuantities:
    def __init__(self, quantities_and_costs):
        self.quantities_and_costs = quantities_and_costs

    def generate_bill_of_quantities(self):
        bill_of_quantities = {}
        for item, (quantity, unit_cost, labor_rate, installation_time) in self.quantities_and_costs.items():
            total_cost = quantity * (unit_cost + labor_rate)
            bill_of_quantities[item] = {
                "Quantity": quantity,
                "Unit Cost": unit_cost,
                "Labor Rate": labor_rate,
                "Installation Time (hrs)": installation_time,
                "Total Cost": total_cost
            }
        return bill_of_quantities

