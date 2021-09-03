# -*- coding: UTF-8 -*-

import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

class Planner():
    """
    MILP Electricity dad bidding formulation.
    """

    def __init__(self, scenarios:dict, prices:dict, x:np.array=None, curtail:bool=True, soc_max:float=1):
        """
        Init the planner.
        """

        self.pv_scenarios = scenarios['PV']  # (MW) (n_s, n_periods)
        self.wind_scenarios = scenarios['W']  # (MW) (n_s, n_periods)
        self.load_scenarios = scenarios['L']  # (MW) (n_s, n_periods)
        self.nb_scenarios = self.pv_scenarios.shape[0]
        self.s_set = range(self.nb_scenarios)
        self.x = x # (MW)
        self.curtail = curtail

        self.period_hours = 1  # (hour)
        self.nb_periods = self.pv_scenarios.shape[1]
        self.t_set = range(self.nb_periods)

        self.dad_prices = prices['dad']  # (euros/MWh)  (n_periods,)
        self.imb_pos_prices = prices['imb +']  # (euros/MWh) (n_periods,)
        self.imb_neg_prices = prices['imb -']  # (euros/MWh) (n_periods,)

        # BESS parameters
        self.soc_max = soc_max
        self.charge_power = self.soc_max / 2
        self.discharge_power = self.soc_max / 2
        self.soc_min = 0
        self.charge_eff = 0.95
        self.discharge_eff = 0.95
        self.soc_ini = 0
        self.soc_end = 0

        self.time_building_model = None
        self.time_solving_model = None

        # Create model
        self.model = self.create_model()

        # Solve model
        self.solver_status = None

    def create_model(self):
        """
        Create the optimization problem.
        """
        t_build = time.time()

        # -------------------------------------------------------------------------------------------------------------
        # 1. create model
        model = gp.Model("planner_dad")

        # -------------------------------------------------------------------------------------------------------------
        # 2. create variables
        # 2.1 First-stage variables -> x = dad bidding
        x = model.addVars(self.nb_periods, lb=-1000, ub=1000, obj=0, vtype=GRB.CONTINUOUS, name="x") # Retailer position (injection > 0, withdrawal < 0) (MWh)
        if self.x is not None:
            for t in self.t_set:
                x[t].setAttr("ub", self.x[t])
                x[t].setAttr("lb", self.x[t])

        # 2.2 Second-stage variables -> y = realisation of the random variables in scenarios omega
        y = model.addVars(self.nb_scenarios, self.nb_periods, lb=-1000, ub=1000, obj=0, vtype=GRB.CONTINUOUS, name="y")  # Retailer position in scenario s (injection > 0, withdrawal < 0) (MWh)
        y_short = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_short") # Retailer position in short in scenario s y_short >=  (x - y) (MWh)
        y_long = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_long")   # Retailer position in long in scenario s y_long  >=  (y - x) (MWh)

        y_PV = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_PV")  # PV generation in scenario s (MW)
        y_W = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_W")  # Wind generation in scenario s (MW)
        y_L = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_L")  # Load generation in scenario s (MW)

        y_s = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_s")  # BESS state of charge in scenario s (MWh)
        y_cha = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_cha")  # BESS charging power in scenario s (MW)
        y_dis = model.addVars(self.nb_scenarios, self.nb_periods, lb=0, ub=GRB.INFINITY, obj=0, vtype=GRB.CONTINUOUS, name="y_dis")  # BESS discharging power in scenario s (MW)
        y_b = model.addVars(self.nb_scenarios, self.nb_periods, obj=0, vtype=GRB.BINARY, name="y_b")  # BESS binary variable to prevent from charging or discharging simultaneously in scenario s (-)

        # -------------------------------------------------------------------------------------------------------------
        # 3. Objective: maximize the IJF_paper profit
        # -------------------------------------------------------------------------------------------------------------
        # Maximization of the expected profit over all scenarios s with an equal probability
        # -> the dad prices are assumed to be equal to the expected value for a given time periocd t: dad_prices_t = E[dad_prices_{t,s}]
        # -> the pos imb prices are assumed to be equal to the expected value: imb_pos_prices_t = E[imb_pos_prices_{t,s}]
        # -> the neg imb prices are assumed to be equal to the expected value: imb_neg_prices_t = E[imb_neg_prices_{t,s}]

        # max sum_t [sum_s alpha_s {self.dad_prices[t] * y[s,t] -(self.imb_neg_prices[t] * y_short[s,t] + self.imb_pos_prices[t] * y_long[s,t])  }]

        dad_profit = gp.quicksum(self.dad_prices[t] * x[t] for t in self.t_set)
        short_penalty = gp.quicksum(gp.quicksum(self.imb_neg_prices[t] * y_short[s,t] for s in self.s_set)/self.nb_scenarios for t in self.t_set)
        long_penalty = gp.quicksum(gp.quicksum(self.imb_pos_prices[t] * y_long[s,t] for s in self.s_set)/self.nb_scenarios for t in self.t_set)

        model.setObjective(dad_profit - (short_penalty + long_penalty), GRB.MAXIMIZE)

        # -------------------------------------------------------------------------------------------------------------
        # 4. create constraints

        # Second-stage constraints
        # Energy balance equation
        model.addConstrs((y[s,t] - self.period_hours * (y_PV[s,t] + y_W[s,t] + y_dis[s,t] - y_cha[s,t]) + y_L[s,t] == 0 for s in self.s_set for t in self.t_set), name='c_balance')

        # Short position cst
        model.addConstrs((y_short[s,t] >= (x[t] - y[s,t]) for s in self.s_set for t in self.t_set), name='c_short')
        # Long position cst
        model.addConstrs((y_long[s,t] >= (y[s,t] - x[t]) for s in self.s_set for t in self.t_set), name='c_long')

        # Generation & load cst
        if self.curtail:
            model.addConstrs((y_PV[s,t] <= self.pv_scenarios[s,t] for s in self.s_set for t in self.t_set), name='c_PV')
            model.addConstrs((y_W[s,t] <= self.wind_scenarios[s,t] for s in self.s_set for t in self.t_set), name='c_W')
        else:
            model.addConstrs((y_PV[s,t] == self.pv_scenarios[s,t] for s in self.s_set for t in self.t_set), name='c_PV')
            model.addConstrs((y_W[s,t] == self.wind_scenarios[s,t] for s in self.s_set for t in self.t_set), name='c_W')
        model.addConstrs((y_L[s,t] == self.load_scenarios[s,t] for s in self.s_set for t in self.t_set), name='c_L')

        # BESS constraints
        # max charge cst
        model.addConstrs((y_cha[s,t] <= y_b[s,t] * self.charge_power for s in self.s_set for t in self.t_set), name='c_max_charge')
        # max discharge cst
        model.addConstrs((y_dis[s,t] <= (1 - y_b[s,t]) * self.discharge_power for s in self.s_set for t in self.t_set), name='c_max_discharge')
        # min soc cst
        model.addConstrs((y_s[s,t] >= self.soc_min for s in self.s_set for t in self.t_set), name='c_min_s')
        # min soc cst
        model.addConstrs((y_s[s,t] <= self.soc_max for s in self.s_set for t in self.t_set), name='c_max_s')

        # BESS dynamics first period
        model.addConstrs((y_s[s,0] - self.period_hours * (self.charge_eff * y_cha[s,0] - y_dis[s,0] / self.discharge_eff) == self.soc_ini for s in self.s_set), name='c_BESS_first_period')
        # BESS dynamics from second to last periods
        model.addConstrs((y_s[s,t] - y_s[s,t-1]- self.period_hours * (self.charge_eff * y_cha[s,t] - y_dis[s,t] / self.discharge_eff) == 0 for s in self.s_set for t in range(1, self.nb_periods)), name='c_BESS_dynamics')
        # BESS dynamics last period
        model.addConstrs((y_s[s, self.nb_periods-1]  == self.soc_end for s in self.s_set), name='c_BESS_last_period')

        # -------------------------------------------------------------------------------------------------------------
        # 5. Store variables
        self.allvar = dict()
        self.allvar['x'] = x
        self.allvar['y'] = y
        self.allvar['y_short'] = y_short
        self.allvar['y_long'] = y_long
        self.allvar['y_PV'] = y_PV
        self.allvar['y_W'] = y_W
        self.allvar['y_L'] = y_L
        self.allvar['y_cha'] = y_cha
        self.allvar['y_dis'] = y_dis
        self.allvar['y_s'] = y_s
        self.allvar['y_b'] = y_b

        self.time_building_model = time.time() - t_build
        # print("Time spent building the mathematical program: %gs" % self.time_building_model)

        return model

    def solve(self, LogToConsole:bool=False, logfile:str="", Threads:int=0, MIPFocus:int=0, TimeLimit:float=GRB.INFINITY):

        t_solve = time.time()

        self.model.setParam('LogToConsole', LogToConsole) # no log in the console if set to False
        # self.model.setParam('OutputFlag', outputflag) # no log into console and log file if set to True
        # self.model.setParam('MIPGap', 0.01)
        self.model.setParam('TimeLimit', TimeLimit)
        self.model.setParam('MIPFocus', MIPFocus)
        # self.model.setParam('DualReductions', 0) # Model was proven to be either infeasible or unbounded. To obtain a more definitive conclusion, set the DualReductions parameter to 0 and reoptimize.

        # If you are more interested in good quality feasible solutions, you can select MIPFocus=1.
        # If you believe the solver is having no trouble finding the optimal solution, and wish to focus more attention on proving optimality, select MIPFocus=2.
        # If the best objective bound is moving very slowly (or not at all), you may want to try MIPFocus=3 to focus on the bound.

        self.model.setParam('LogFile', logfile) # no log in file if set to ""
        self.model.setParam('Threads', Threads) # Default value = 0 -> use all threads

        self.model.optimize()
        self.solver_status = self.model.status
        self.time_solving_model = time.time() - t_solve

    def store_solution(self):

        m = self.model

        solution = dict()
        solution['status'] = m.status
        if solution['status'] == 2 or solution['status'] == 9:
            solution['obj'] = m.objVal

            # 1 dimensional variables
            for var in ['x']:
                solution[var] = [self.allvar[var][t].X for t in self.t_set]

            # 2 dimensional variables
            for var in ['y', 'y_short', 'y_long', 'y_PV', 'y_W', 'y_L', 'y_dis', 'y_cha', 'y_s', 'y_b']:
                solution[var] = [[self.allvar[var][s, t].X for t in self.t_set] for s in self.s_set]

            solution['dad_profit'] = sum([solution['x'][t] * self.dad_prices[t] for t in self.t_set])
            solution['short_penalty'] = sum([sum([solution['y_short'][s][t] * self.imb_neg_prices[t] for t in self.t_set]) for s in self.s_set]) / self.nb_scenarios
            solution['long_penalty'] = sum([sum([solution['y_long'][s][t] * self.imb_pos_prices[t] for t in self.t_set]) for s in self.s_set]) / self.nb_scenarios
            solution['obj2'] = solution['dad_profit'] - (solution['short_penalty'] + solution['long_penalty'])
        else:
            print('WARNING model is not OPTIMAL')
            solution['obj'] = math.nan

        # 3. Timing indicators
        solution["time_building"] = self.time_building_model
        solution["time_solving"] = self.time_solving_model
        solution["time_total"] = self.time_building_model + self.time_solving_model

        return solution

    def export_model(self, filename):
        """
        Export the pyomo model into a cpxlp format.
        :param filename: directory and filename of the exported model.
        """

        self.model.write("%s.lp" % filename)
        # self.model.write("%s.mps" % filename)


if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())

    dir_path = '../../../elia_case_study/bidding/export/dad_bidding/'
    if not os.path.isdir(dir_path):  # test if directory exist
        os.makedirs(dir_path)

    soc_max = 500

    q_pos = 2
    q_neg = 2

    dad_price = 100  # euros /MWh
    # pos_imb = neg_imb = q * dad_price # euros /MWh
    pos_imb = q_pos * dad_price  # euros /MWh
    neg_imb = q_neg * dad_price  # euros /MWh
    gamma = (dad_price + pos_imb) / (pos_imb + neg_imb)
    print('dad_price %s pos_imb %s neg_imb %s GAMMA %s' % (dad_price, pos_imb, neg_imb, gamma))

    # load data
    df_gen = pd.read_csv('../../../elia_case_study/data/generation.csv', parse_dates=True, index_col=0)
    df_load = pd.read_csv('../../../elia_case_study/data/load.csv', parse_dates=True, index_col=0)
    df_dad = pd.read_csv('../../../elia_case_study/data/dad.csv', parse_dates=True, index_col=0)
    df_imb = pd.read_csv('../../../elia_case_study/data/imb.csv', parse_dates=True, index_col=0)

    nb_scenarios = 5
    pv = df_gen['PV true']['2020-1-1':'2020-1-'+str(nb_scenarios)]
    wind = df_gen['W on true']['2020-1-1':'2020-1-' + str(nb_scenarios)]
    load = 0.05 * df_load['load true']['2020-1-1':'2020-1-' + str(nb_scenarios)]

    # 20 scenarios
    scenarios = dict()
    scenarios['PV'] = pv.values.reshape(nb_scenarios,24)
    scenarios['W'] = wind.values.reshape(nb_scenarios, 24)
    scenarios['L'] = load.values.reshape(nb_scenarios, 24)

    # Plot point forecasts vs observations
    FONTSIZE = 10
    plt.figure()
    net = pv.values + wind.values - load.values
    plt.plot(pv.values, label='PV')
    plt.plot(wind.values, label='W on')
    plt.plot(load.values, label='Load')
    plt.plot(net, 'r', label='net')
    plt.ylabel('MW', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()

    prices = dict()
    prices['dad'] = np.asarray([dad_price]*16+[3*dad_price]*4+[dad_price]*4)
    prices['imb +'] = np.asarray([pos_imb]*16+[3*pos_imb]*4+[pos_imb]*4)
    prices['imb -'] = np.asarray([neg_imb]*16+[3*neg_imb]*4+[neg_imb]*4)

    # prices = dict()
    # prices['dad'] = df_dad['2020-1-3'].values.reshape(-1)
    # prices['imb +'] = q_pos * df_dad['2020-1-3'].values.reshape(-1)
    # prices['imb -'] = q_neg * df_dad['2020-1-3'].values.reshape(-1)

    plt.figure()
    plt.plot(prices['dad'], label='dad')
    plt.plot(prices['imb -'] , label='imb neg')
    plt.ylabel('€/MWh', fontsize=FONTSIZE, rotation='horizontal')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()

    # Dad planner
    planner = Planner(scenarios=scenarios, prices=prices, soc_max=soc_max)
    planner.export_model(dir_path + 'planner_dad')
    planner.solve()
    sol_planner = planner.store_solution()
    #
    print('profit %.2f k€ dad bid %.2f short penalty %.2f k€ long penalty %.2f k€' % (sol_planner['obj'] / 1000, sol_planner['dad_profit'] / 1000, sol_planner['short_penalty'] / 1000, sol_planner['long_penalty'] / 1000))

    plt.figure()
    plt.plot(sol_planner['x'], 'r', label='planning')
    for s in range(nb_scenarios):
        # plt.plot(solution['y'][0], 'k',label='position')
        net = scenarios['PV'][s] + scenarios['W'][s] - scenarios['L'][s]
        # plt.plot(scenarios['PV'][s] + scenarios['W'][s], 'gray')
        # plt.plot(scenarios['L'][s], 'b')
        # plt.plot(solution['y_s'][0], 'orange',label='y_s')
        plt.plot(net, 'g', label='net=generation-load')
    plt.ylabel('MW', fontsize=FONTSIZE, rotation='horizontal')
    plt.ylim(-1000,2000)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()

    # Economic dispatch based on actual realization of uncertainties
    np.random.seed(seed=0)
    # omega = np.random.randint(nb_scenarios, size=nb_scenarios)
    omega = range(nb_scenarios)
    res_O = []
    res_planner = []
    for s in omega:
        # pick a scenario
        scenarios_dispatch = dict()
        scenarios_dispatch['PV'] = scenarios['PV'][s,:].reshape(1, 24)
        scenarios_dispatch['W'] = scenarios['W'][s,:].reshape(1, 24)
        scenarios_dispatch['L'] = scenarios['L'][s,:].reshape(1, 24)

        # oracle
        oracle = Planner(scenarios=scenarios_dispatch, prices=prices, soc_max=soc_max)
        oracle.export_model(dir_path + 'planner_dad')
        oracle.solve()
        sol_oracle = oracle.store_solution()

        res_O.append(sol_oracle['obj'] / 1000)

        dispatch = Planner(scenarios=scenarios_dispatch, prices=prices, x=sol_planner['x'], soc_max=soc_max)
        dispatch.export_model(dir_path + 'dispatch')
        dispatch.solve()
        sol_dispatch = dispatch.store_solution()
        res_planner.append(sol_dispatch['obj'] / 1000)

        print('s %s net %.2f k€ dad bid %.2f short penalty %.2f k€ long penalty %.2f k€' % (s, sol_dispatch['obj'] / 1000, sol_dispatch['dad_profit'] / 1000, sol_dispatch['short_penalty'] / 1000, sol_dispatch['long_penalty'] / 1000))
        print('oracle net %.2f k€ dad bid %.2f short penalty %.2f k€ long penalty %.2f k€' % (sol_oracle['obj'] / 1000, sol_oracle['dad_profit'] / 1000, sol_oracle['short_penalty'] / 1000, sol_oracle['long_penalty'] / 1000))
        net = scenarios_dispatch['PV'][0] + scenarios_dispatch['W'][0] - scenarios_dispatch['L'][0]

        # plt.figure()
        # plt.plot(sol_oracle['x'], 'b', label='x oracle')
        # plt.plot(sol_planner['x'], 'r', label='planning')
        # plt.ylabel('MW', fontsize=FONTSIZE, rotation='horizontal')
        # plt.ylim(-1000,2000)
        # plt.title(str(s) + ' oracle vs planner')
        # plt.xticks(fontsize=FONTSIZE)
        # plt.yticks(fontsize=FONTSIZE)
        # plt.legend(fontsize=FONTSIZE)
        # plt.tight_layout()
        # plt.show()
        #
        # plt.figure()
        # plt.plot(sol_planner['x'], 'r', label='planning')
        # plt.plot(sol_dispatch['y'][0], 'k',label='position')
        # plt.plot(sol_dispatch['y_s'][0], 'orange',label='y_s')
        # # plt.plot(scenarios_dispatch['PV'][0] + scenarios_dispatch['W'][0], 'gray', label='generation')
        # # plt.plot(scenarios_dispatch['L'][0], 'b', label='load')
        # plt.plot(net, 'g', label='net=generation-load')
        # plt.ylabel('MW', fontsize=FONTSIZE, rotation='horizontal')
        # plt.ylim(-1000,2000)
        # plt.title(s)
        # plt.xticks(fontsize=FONTSIZE)
        # plt.yticks(fontsize=FONTSIZE)
        # plt.legend(fontsize=FONTSIZE)
        # plt.tight_layout()
        # plt.show()

    plt.figure()
    plt.plot(res_O, 'r', label='oracle')
    plt.plot(res_planner, 'k', label='planner')
    plt.ylabel('k€', fontsize=FONTSIZE, rotation='horizontal')
    plt.xlabel('scenarios', fontsize=FONTSIZE, rotation='horizontal')
    plt.ylim(-1000, 1200)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tight_layout()
    plt.show()

    print('O %.2f planner %.2f' %(sum(res_O), sum(res_planner)))