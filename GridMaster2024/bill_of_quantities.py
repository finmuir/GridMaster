class BillOfQuantities:
    def __init__(self, quantities_and_costs, labor_cost, through_poles=0):
        self.quantities_and_costs = quantities_and_costs
        self.labor_cost = labor_cost
        self.through_poles = through_poles

    def generate_bill_of_quantities(self):
        bill_of_quantities = {}
        for item, (quantity, unit_cost, installation_time) in self.quantities_and_costs.items():
            if item == "Bobbin Insulators":
                # Adjusting quantity based on through poles
                quantity += (2 * self.through_poles)
            total_cost = (quantity * unit_cost) + (installation_time * self.labor_cost)
            bill_of_quantities[item] = {
                "Quantity": quantity,
                "Unit Cost": unit_cost,
                "Labor Cost": self.labor_cost,
                "Installation Time (hrs)": installation_time,
                "Total Cost": total_cost
            }
        return bill_of_quantities
