"""

    Generation Sizer for "Energy 4 Development" VIP

    Code by Alfredo Scalera (alfredo.scalera.2019@uni.strath.ac.uk)

    Based on MATLAB code by Steven Nolan ( ).


"""


import random
import math
from PVoutput2 import PVOutput

import os
import plotly.graph_objs as go
import plotly.io as pio

class Particle:
    # technical trackers
    fuel_used = 0
    Edump = 0

    # previous position placeholders
    prev_pos = []
    prev_fuel = 0

    # PSO variables
    pbest_pos = []
    pbest_value = 0
    gbest_pos = []
    gbest_value = 0

    cost = 0

    autonomDays = 0

    def __init__(self, name, array_len):
        """
        Particle object used for PSO in GenSizer object.

        Parameters
        ----------
        name : str
            Name of particle.
        array_len : int
            Length of arrays dedicated to power generation and
            batteries. GenSizer object should provide this.

        """

        self.name = name

        self.Psol = [0] * array_len
        self.Pgen = [0] * array_len
        self.Ebatt = [0] * (array_len + 1)  # to avoid overshooting

        # random x, y, z coordinates - changed to have at least 50 of each component to ensure it always runs
        self.pos = [random.randint(50, 500), random.randint(50, 500), random.randint(50, 500)]

        # initial velocity is zero in all directions
        self.vel = [0, 0, 0]

        self.gbest_value = 10 ** 60
        self.pbest_value = 10 ** 60
        self.cost = 10 ** 60
        self.pbest_pos = self.pos.copy()

    def __str__(self):
        """
        Prints name, position, velocity and cost value of particle.
        Used for debugging purposes.

        Returns
        -------
        str
            String containing name, position, velocity,
            cost of particle.

        """

        return (self.name + ", Position: " + str(self.pos) + ", Velocity: "
                + str(self.vel) + ". Cost: " + str(self.cost))

    def updatePosition(self):
        """
        Updates posistion of particle based on velocity.
        Saves previous location for later use.

        """

        # save previous position
        self.prev_pos = self.pos.copy()

        self.pos[0] = self.pos[0] + self.vel[0]  # update x
        self.pos[1] = self.pos[1] + self.vel[1]  # update y
        self.pos[2] = self.pos[2] + self.vel[2]  # update z



class GenSizer:

    def __init__(self, swarm_size, power_demand, psol_unit,
                 sol_cost, batt_cost, gen_cost, fuel_cost,  # costs
                 batt_Wh_max_unit, batt_Wh_min_unit,  # battery parameters
                 gen_max_power_out, gen_fuel_req,  # generator parameters
                 max_off_hours, min_autonomy_days):
        """
        Establishes optimal combination of PV array size, number of
        batteries, and number of generators needed to meet power
        mini-grid's power demand with lowest cost. Uses particle swarm
        optimisation to do so.

        Parameters
        ----------
        swarm_size : int
            Number of particles used in PSO.
        power_demand : array-like
            Array containing yearly mini-grid power demand.
            Hourly timestep (length should be 8760).
        psol_unit : array-like
            Array containing power provided by single PV panel.
        sol_cost : float
            Cost of single PV panel.
        batt_cost : float
            Cost of single battery.
        gen_cost : float
            Cost of single diesel generator.
        fuel_cost : float
            Cost of fuel per liter.
        batt_Wh_max_unit : float
            Battery maximum Wh capacity.
        batt_Wh_min_unit : float
            Battery minimum Wh capacity (lowest it can go).
        gen_max_power_out : float
            Maximum power output of generator in Watts.
        gen_fuel_req : float
            Fuel requirement in liter per hour of generation
            at max power.
        max_off_hours : int
            Maximum hours in a year the grid can be offline for.
            More will decrease cost.
        min_autonomy_days : int
            Number of autonomy days required for grid. More will
            increase cost.

        """

        self.swarm_size = int(swarm_size)
        self.Pdem = power_demand
        print(self.Pdem)
        self.Psol_unit = psol_unit

        # component costs
        self.solCost = sol_cost
        self.battCost = batt_cost
        self.genCost = gen_cost
        self.fuelCost = fuel_cost

        # technical parameters
        self.EbattMax_unit = batt_Wh_max_unit
        self.EbattMin_unit = batt_Wh_min_unit
        self.Pgen_unit = gen_max_power_out
        self.fuelReq = gen_fuel_req

        # offline & days of autonomy
        self.timebreakerMax = max_off_hours
        self.autonomDaysMin = min_autonomy_days

        # generate swarm
        self.swarm = []
        array_len = len(power_demand)
        for i in range(self.swarm_size):
            name = "Particle " + str(i)
            self.swarm.append(Particle(name, array_len))

        self.Pdem = power_demand
        self.Psol_unit = psol_unit

        # naughty list
        self.invalid_particles = []

        # Initialize your GenSizer object with parameters
        self.batt_Wh_max_unit = batt_Wh_max_unit
        self.batt_Wh_min_unit = batt_Wh_min_unit


    def _test_constraints(self):
        """
        Checks which particles do not meet power demand. Adds invalid
        particles to "naughty" list so they can be dealt with later.

        """

        self.invalid_particles.clear()

        for p in self.swarm:  # p = particle

            p.prev_fuel = p.fuel_used
            p.fuel_used = 0

            # check if particle has negative values
            if p.pos[0] < 0 or p.pos[1] < 0 or p.pos[2] < 0:
                self.invalid_particles.append(p)
                continue

            p.Pgen = [0] * len(self.Pdem)
            p.Ebatt = [0] * (len(self.Pdem) + 1)  # +1 avoids overshoot
            p.Edump = 0


            #number of solar pannels
            Ns = p.pos[0]
            #number of batteries
            Nb = p.pos[1]
            #number of generators
            Ng = p.pos[2]

            EbattMin = Nb * self.EbattMin_unit
            EbattMax = Nb * self.EbattMax_unit
            Pgen = Ng * self.Pgen_unit

            # assume batteries initally fully charged
            p.Ebatt[0] = EbattMax


            # avg power needed for 1 day
            P1day = sum(self.Pdem[0:24])

            # check if configuration can sustain microgrid for set days
            p.autonomDays = Nb * (self.EbattMax_unit - self.EbattMin_unit) / P1day
            if p.autonomDays < self.autonomDaysMin:
                self.invalid_particles.append(p)
                continue

            timebreaker = 0

            for t in range(len(self.Pdem)):

                p.Psol[t] = Ns * self.Psol_unit[t]

                # solar power matches demand
                p.Ebatt[t + 1] = p.Ebatt[t]

                # # solar power exceeds demand, charge batteries


                if p.Psol[t] > self.Pdem[t]:
                    Echarge = p.Psol[t] - self.Pdem[t]
                    # Echarge = Pcharge * 1      1Wh = 1W*1hr

                    # energy charge exceeds max capacity, dump excess energy
                    if (p.Ebatt[t] + Echarge) > EbattMax:
                        p.Ebatt[t + 1] = EbattMax
                        p.Edump += (p.Ebatt[t] + Echarge - EbattMax)
                        #Echarge=EbattMax-p.Ebatt[t]

                    # energy charge below max capacity, charge battery
                    else:
                        p.Ebatt[t + 1] = p.Ebatt[t] + Echarge

                # solar power below demand
                elif p.Psol[t] < self.Pdem[t]:
                    Edisch = self.Pdem[t] - p.Psol[t]
                    # Edisch = Pdisch * 1         1Wh = 1W*1hr

                    # battery energy enough to meet demand
                    if (p.Ebatt[t] - Edisch) >= EbattMin:
                        p.Ebatt[t + 1] = p.Ebatt[t] - Edisch

                    # battery energy below demand, activate generators
                    else:
                        Edisch = p.Ebatt[t] - EbattMin
                        Echarge =  Edisch + p.Psol[t] + Pgen - self.Pdem[t]
                        p.Ebatt[t + 1] = Echarge+EbattMin


                        p.Pgen[t] = Pgen
                        p.fuel_used += (Ng * self.fuelReq)

                        # generator power below demand
                        if p.Ebatt[t + 1] < EbattMin:
                            self.invalid_particles.append(p)
                            timebreaker += 1
                            if timebreaker > self.timebreakerMax:
                                self.invalid_particles.append(p)
                                break

                        # generator exceeds demand, charge batteries
                        elif p.Ebatt[t + 1] > EbattMax:
                            p.Edump += (p.Ebatt[t + 1] - EbattMax)
                            p.Ebatt[t + 1] = EbattMax
    def _delete_invalid(self):
        """
        Deletes invalid particles. Only used in initialisation phase
        of particle swarm optimisation.

        """

        for p in self.invalid_particles:
            self.swarm.remove(p)

    def _update_pos_all(self):
        """
        Updates position of all particles within swarm.

        """

        for p in self.swarm:
            p.updatePosition()

    def _reset_invalid(self):
        """
        Resets to last valid position all particles in "naughty" list.

        """

        for p in self.swarm:
            if p in self.invalid_particles:
                p.pos = p.prev_pos.copy()
                p.vel = [0, 0, 0]
                p.fuel_used = p.prev_fuel

    def _fitness_all(self):
        """
        Evaluates cost of each particle within swarm, updates global
        best value and position, updates particle's best value and
        position.

        """

        # evaluate cost (obj function)
        for p in self.swarm:
            Ns = p.pos[0]
            Nb = p.pos[1]
            Ng = p.pos[2]

            p.cost = Ns * self.solCost * 1.01 + Nb * 3 * self.battCost * 1.01 + Ng * self.genCost * 1.1 * 4.5 + 1.5 * p.fuel_used * self.fuelCost

            # update particle pbest
            if p.cost < p.pbest_value:
                p.pbest_pos = p.pos.copy()
                p.pbest_value = p.cost

        # update gbest for all particles
        values = []
        for p in self.swarm:
            values.append(p.pbest_value)
        # find gbest and associated particle
        gbest = min(values)
        i = values.index(gbest)
        gbest_pos = self.swarm[i].pbest_pos.copy()

        # update gbest for each particle in swarm
        for p in self.swarm:
            p.gbest_pos = gbest_pos  # .copy()
            p.gbest_value = gbest

    def _update_vel_all(self, current_iter,max_iter):
        """
        Updates velocity for all particles in swarm.

        Parameters
        ----------
        current_iter : int
            Number of PSO loop iteration. Used for dynamic
            intertia (w), self-confidence (c1) and conformity (c2)
            hyperparameters.

        """

        for p in self.swarm:
            # PSO parameters
            # w inertia
            # c1 self confidence, c2 social conformity
            # r1, r2 random factors


            w = 0.5*(self.max_iter - current_iter)/(self.max_iter) + 0.4 #inertia allows for high exploration at the start of pso

            c1 = 2.05  # particle's self confidence
            c2 = 2.05  # particle's conformity


            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            r4 = random.random()
            r5 = random.random()
            r6 = random.random()

            pbest = p.pbest_pos.copy()
            gbest = p.gbest_pos.copy()

            p.vel[0] = math.floor(w * p.vel[0] + c1 * r1 * (pbest[0] - p.pos[0])
                                  + c2 * r2 * (gbest[0] - p.pos[0]))
            p.vel[1] = math.floor(w * p.vel[1] + c1 * r3 * (pbest[1] - p.pos[1])
                                  + c2 * r4 * (gbest[1] - p.pos[1]))
            p.vel[2] = math.floor(w * p.vel[2] + c1 * r5 * (pbest[2] - p.pos[2])
                                  + c2 * r6 * (gbest[2] - p.pos[2]))


    def _check_converge(self):
        """
        Checks if swarm has converged before maximum number of
        itarations met.

        """

        positions = [particle.pos for particle in self.swarm]

        velocities = [particle.vel for particle in self.swarm]

        # self.converged is true if:
        #   all the particles have 0 velocity in all directions
        #   (and condition to check for empty list)
        #   all particles are in the same position
        self.converged = ((velocities.count([0, 0, 0]) == len(velocities))
                          and (positions.count(positions[0]) == len(positions)))

        # cleanup
        del velocities
        del positions

    def battery_power(self):
        discharge = [0] * 8760
        charge = [0] * 8760
        energy_difference = [0] * 8760
        dump = [0] * 8760

        #  particle parameters
        Ns = self.swarm[0].pos[0]
        Nb = self.swarm[0].pos[1]
        Ng = self.swarm[0].pos[2]
        EbattMin = Nb * self.EbattMin_unit
        EbattMax = Nb * self.EbattMax_unit
        Pgen = Ng * self.Pgen_unit

        # Initialize battery energy
        Ebatt = [EbattMax]

        # Calculate Psol
        Psol = [Ns * unit for unit in self.Psol_unit]

        for t in range(len(self.Pdem)):
            # Solar power matches demand
            if Psol[t] == self.Pdem[t]:
                Ebatt.append(Ebatt[-1])  # Maintain the same battery energy
            # Solar power exceeds demand, charge batteries
            elif Psol[t] > self.Pdem[t]:
                charge[t] = Psol[t] - self.Pdem[t]
                # Energy charge exceeds max capacity, dump excess energy
                if (Ebatt[t] + charge[t]) > EbattMax:
                    dump[t] = (Ebatt[t] + charge[t]) - EbattMax
                    Ebatt.append(EbattMax)
                else:
                    Ebatt.append(Ebatt[t] + charge[t])
            # Solar power below demand
            else:
                discharge[t] = self.Pdem[t] - Psol[t]
                # Battery energy enough to meet demand
                if (Ebatt[t] - discharge[t]) >= EbattMin:
                    Ebatt.append(Ebatt[t] - discharge[t])
                # Battery energy below demand, activate generators
                else:
                    discharge[t] = Ebatt[t] - EbattMin
                    charge[t] =   Psol[t] + Pgen - self.Pdem[t]- discharge[t]
                    Ebatt.append(charge[t] + EbattMin)
                    dump[t] = (Ebatt[t + 1] - EbattMax) if Ebatt[t + 1] > EbattMax else 0

        # Check if battery energy exceeds max capacity
        for t in range(len(self.Pdem)):
            if Ebatt[t + 1] > EbattMax:
                charge[t] = EbattMax - Ebatt[t]
                Ebatt[t + 1] = EbattMax
            energy_difference[t] = charge[t] - discharge[t]

        for t in range(len(self.Pdem)):
            if Nb == 0:
                discharge[t] = 0
                charge[t] = 0

        self.power_discharge = discharge
        self.power_charge = charge
        self.dump_energy = dump




    def optimise(self, max_iter, final_plot=False, animate=False):
        """
        Optimises generation mix by applying PSO.

        Parameters
        ----------
        max_iter : int
            maximum number of iterations.
        final_plot : bool, optional
            Plots power demand (W), solar power (W),
            battery energy (Wh) and generator power (W)
            against time (h). The default is False.
        animate : bool, optional
            Animate swarm while optimising. The default is False.

        """

        # used for inertia correction (w) in velocity update
        self.max_iter = max_iter

        # remove particles outside feasible region
        self._test_constraints()
        self._delete_invalid()

        # PSO loop
        self.converged = False
        i = 0
        while (i < max_iter) and (self.converged == False):

            if i % 1 == 0:
                print("\n\niteration:", i + 1)

                print("\nPos:", self.swarm[0].pos)
                print("Cost:", self.swarm[0].cost)
                print("Vel:", self.swarm[0].vel)
                print("FUEL:", self.swarm[0].fuel_used)

                print("\nPbest:", self.swarm[0].pbest_pos)
                print("Pb cost:", self.swarm[0].pbest_value)

                print("\nGbest:", self.swarm[0].gbest_pos)
                print("Gb cost:", self.swarm[0].gbest_value)

            self._update_pos_all()
            self._test_constraints()
            self._reset_invalid()
            self._fitness_all()
            self._update_vel_all(i,max_iter)
            self._check_converge()
            i = i+1


        self.total_cost = self.swarm[0].cost
        self.battery_power()

    def plot_graphs(self):

        Num_solar = self.swarm[0].pos[0]
        num_batteries = self.swarm[0].pos[1]
        num_generator = self.swarm[0].pos[2]
        fuel_used = self.swarm[0].fuel_used
        Cost = round(self.swarm[0].cost, 2)
        autonomDays = round(self.swarm[0].autonomDays, 2)


        # t = [x for x in range(72)]
        # xmax = 72
        #
        power_demand = self.Pdem
        power_battery_discharge = self.power_discharge
        power_battery_charge = self.power_charge
        power_generator = self.swarm[0].Pgen
        power_solar = self.swarm[0].Psol
        #
        #
        EbattMin = num_batteries * self.EbattMin_unit
        EbattMax = num_batteries * self.EbattMax_unit
        dumped_energy = self.dump_energy
        #
        # # Create directory if it doesn't exist
        # os.makedirs('static/plots/', exist_ok=True)
        #
        # # Plot power supply and demand against time
        # fig_supply_demand = go.Figure()
        # fig_supply_demand.add_trace(
        #     go.Scatter(x=t, y=power_demand[:len(t)], mode='lines', name='Power Demand', line=dict(color='black')))
        # fig_supply_demand.add_trace(
        #     go.Scatter(x=t, y=power_battery_discharge[:len(t)], mode='lines', name='Battery Discharge Power',
        #                line=dict(color='blue', dash='dash')))
        # fig_supply_demand.add_trace(
        #     go.Scatter(x=t, y=power_battery_charge[:len(t)], mode='lines', name='Battery Charge Power',
        #                line=dict(color='orange', dash='dashdot')))
        # fig_supply_demand.add_trace(
        #     go.Scatter(x=t, y=power_generator[:len(t)], mode='lines', name='Power from Generators',
        #                line=dict(color='red', dash='dashdot')))
        # fig_supply_demand.add_trace(
        #     go.Scatter(x=t, y=power_solar[:len(t)], mode='lines', name='Power from Solar Panels',
        #                line=dict(color='green', dash='dot')))
        # fig_supply_demand.update_layout(title='Power Supply and Demand vs Time', xaxis_title='Time (h)',
        #                                 yaxis_title='Power (W)')
        # plot_file_supply_demand = 'static/plots/power_supply_demand.html'
        # pio.write_html(fig_supply_demand, file=plot_file_supply_demand, auto_open=False)
        #
        # #pio.show(fig_supply_demand)
        #
        # # Extract battery energy data from the swarm object
        battery_energy_data = self.swarm[0].Ebatt[0:8760]
        #
        # if num_batteries >= 0:
        #     # Create a Plotly figure for battery energy
        #     fig_battery_energy = go.Figure()
        #     fig_battery_energy.add_trace(
        #         go.Scatter(x=t, y=battery_energy_data, mode='lines', name='Energy stored', line=dict(color='blue')))
        #
        #     # Add horizontal lines for maximum and minimum battery energy levels
        #     fig_battery_energy.add_shape(type="line", x0=0, y0=EbattMax, x1=xmax, y1=EbattMax,
        #                                  line=dict(color="green", width=2, dash="dash"), name="Max Battery Energy")
        #     fig_battery_energy.add_shape(type="line", x0=0, y0=EbattMin, x1=xmax, y1=EbattMin,
        #                                  line=dict(color="red", width=2, dash="dash"), name="Min Battery Energy")
        #
        #     # Update layout of the battery energy plot
        #     fig_battery_energy.update_layout(title='Battery Energy vs Time', xaxis_title='Time (h)',
        #                                      yaxis_title='Battery Energy (Wh)')
        #
        #     # Create directory if it doesn't exist
        #     os.makedirs('static/plots/', exist_ok=True)
        #
        #     # Write the battery energy figure to an HTML file
        #     plot_file_battery_energy = 'static/plots/battery_energy.html'
        #     pio.write_html(fig_battery_energy, file=plot_file_battery_energy, auto_open=False)
        #
        #     #pio.show(fig_battery_energy)
        #
        #     # Plot dumped energy against time
        #     fig_dumped_energy = go.Figure()
        #     fig_dumped_energy.add_trace(go.Scatter(x=t, y=dumped_energy[:len(t)], mode='lines', name='Dumped Energy',
        #                                            line=dict(color='blue')))
        #     fig_dumped_energy.update_layout(title='Wasted Energy vs Time', xaxis_title='Time (h)',
        #                                     yaxis_title='Wasted Energy (Wh)')
        #     plot_file_dumped_energy = 'static/plots/dumped_energy.html'
        #     pio.write_html(fig_dumped_energy, file=plot_file_dumped_energy, auto_open=False)

            #pio.show(fig_dumped_energy)

        return Num_solar, num_batteries, num_generator, fuel_used, Cost, autonomDays,power_demand,power_battery_discharge,power_battery_charge,power_generator,power_solar ,EbattMin,EbattMax ,dumped_energy, battery_energy_data





# # #Placeholder values, replace with actual data need research to find real values and take values from network dessigner
# swarm_size = 20
# # power_demand=[12500]*8760
#
# power_demand=[3000,3000,3000,5000,5000,5000,5000,5000,5000,7500,10000,10000,10000,10000,10000,10000,10000,9000,12000,12000,1200,5000,5000]*365
#
# # # Example: Hourly power demand for a year(estimate profile of demand eg make a full day profile make sure array is same length as pvoutput 8760 hours also does not account for losses of panel )
# sol_cost = 1000            # Example: Cost of a single PV panel(input)
# batt_cost = 1000               # Example: Cost of a single battery(input or prereq)
# gen_cost =  1500              # Example: Cost of a single diesel generator()
# fuel_cost = 1.5                # Example: Cost of fuel per liter (can change so probably input eg if fuel is hard to import or bought in bulk)
# batt_Wh_max_unit = 1000       # Example: Battery maximum Wh capacity(input depeneds on battery)
# batt_Wh_min_unit = 100        # Example: Battery minimum Wh capacity(input depends on battery)
# gen_max_power_out = 1000      # Example: Maximum power output of a generator()
# gen_fuel_req = 1              # Example: Fuel requirement per hour of generation(depends on generator input)
# max_off_hours = 24             # Example: Maximum hours the grid can be offline(limit set by user)
# min_autonomy_days = 0.25         # Example: Minimum number of autonomy days required()
#
# #Example usage
# latitude = -14.24580667  # Latitude of Mthembanji source
# longitude = 34.60600833  # Longitude of Mthembanji source
# panel_capacity = 5000  # Capacity of PV panel in Watts
# year = 2022
# # Year of data(most up to date year)
#
#
# pv_subsystem = PVOutput(latitude, longitude, panel_capacity, year=year)
# output = pv_subsystem.pv_output()
# psol_unit = output     # Example: Hourly power provided by a single PV panel (come from pv output)
# print(psol_unit)
# #Instantiate GenSizer with all required parameters
# gen_sizer = GenSizer(swarm_size, power_demand, psol_unit,
#                     sol_cost, batt_cost, gen_cost, fuel_cost,
#                     batt_Wh_max_unit, batt_Wh_min_unit,
#                     gen_max_power_out, gen_fuel_req,
#                     max_off_hours, min_autonomy_days)
#
#
#
#
# #Perform optimization
# max_iter = 100  # Example: Maximum number of iterations
# gen_sizer.optimise(max_iter)
# gen_sizer.plot_graphs()
