
from pyomo.environ import *
import random
import numpy as np
import matplotlib.pyplot as plt

#
# Model
#

model = AbstractModel()

#
# Parameters
#

model.T= Set()

model.i = Set()

model.capacity = Param(model.i, within=PositiveReals)

model.power = Param(model.i, within=PositiveReals)

model.l_SOC = Param(model.i, within=PositiveReals)

model.u_SOC = Param(model.i, within=PositiveReals)

model.efficiency_c = Param(model.i, within=PositiveReals)

model.efficiency_d = Param(model.i, within=PositiveReals)

model.l_pred = Param(model.T)

#model.l_act = Param(model.T)

model.p_da_pred = Param(model.T, within=PositiveReals)

model.gradient = Param(within=PositiveReals)

model.alpha = Param(within=PositiveReals)

model.beta = Param(within=PositiveReals)

model.uncertainty = Param(within=PositiveReals)

price_scenarios = 50

load_scenarios = 20

#
# Variables
#

# Energy Storage Variable

def capacity_bounds_rule(model, t, i):
    return (model.l_SOC[i] * model.capacity[i], model.u_SOC[i] * model.capacity[i])

model.x = Var(model.T, model.i, bounds=capacity_bounds_rule)

def power_bounds_rule(model, t, i):
    return (0, model.power[i])

model.c = Var(model.T, model.i, bounds=power_bounds_rule)

model.d = Var(model.T, model.i, bounds=power_bounds_rule)

model.u_sch = Var(model.T)

model.l_act = Var(model.T)

# Price Variables

model.p_pm_pred = Var(model.T)

model.p_da_act = Var(model.T)

model.prices = Var(range(price_scenarios), range(load_scenarios))

model.prices_rt = Var(range(price_scenarios), range(load_scenarios))

model.total_prices = Var(range(price_scenarios), range(load_scenarios))

model.pred_cost = Var()

model.prices_no_storage = Var(range(price_scenarios), range(load_scenarios))

model.no_storage_cost = Var()

model.da_pred_prices = Var(range(price_scenarios), model.T)

# Load Variables

model.load_scenarios = Var(range(load_scenarios), model.T)

model.load_scenarios_no_storage = Var(range(load_scenarios), model.T)

model.load_diff_rt = Var(range(load_scenarios), model.T)

# CVaR Variables

model.cvar = Var()

model.z = Var()

model.y = Var(range(price_scenarios), range(load_scenarios))


#
# Constraints
#


# Storage Constraints


def initial_soc_rule(model, i):
    return model.x[23,i] == model.l_SOC[i] * model.capacity[i]

model.initial_soc = Constraint(model.i, rule=initial_soc_rule)

def soc_rule(model, t, i):
    if t == 0:
        s = 23
    else:
        s = t - 1
    return model.x[t,i] == model.x[s,i] + model.efficiency_c[i] * model.c[t,i] - model.d[t,i]

model.soc_constraint = Constraint(model.T, model.i, rule=soc_rule)

def u_scheduling_rule(model, t):
    return model.u_sch[t] == (model.l_pred[t]/1000) + sum(model.c[t,i] - model.efficiency_d[i] *
                                                          model.d[t,i] for i in model.i)

model.u_scheduling_constraint = Constraint(model.T, rule=u_scheduling_rule)

def p_pm_rule(model, t):
    return model.p_pm_pred[t] == model.p_da_pred[t] + model.gradient * model.u_sch[t]

model.p_pm_constraint = Constraint(model.T, rule=p_pm_rule)


# Price Prediction, Scenario Matrix and Actual Price Scenario

def random_price_matrix_rule(model, s, t):
    return model.p_da_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty) \
           + random.uniform(- model.uncertainty, model.uncertainty)

model.price_scenarios = Param(range(price_scenarios), model.T, initialize=random_price_matrix_rule)

def get_da_pred_prices_rule(model, s, t):
    return model.da_pred_prices[s,t] == model.price_scenarios[s,t]

model.get_da_pred_prices = Constraint(range(price_scenarios), model.T, rule=get_da_pred_prices_rule)

def real_time_price_rule(model, t):
    return model.p_da_pred[t] * random.uniform(0.5, 1.5) + \
           random.uniform( - 0.1, 0.1 )  * model.p_da_pred[t]

model.p_rt = Param(model.T, initialize=real_time_price_rule)

def real_time_price_matrix_rule(model, s, t):
    return model.p_da_pred[t] * random.uniform(0.5, 1.5) + \
           random.uniform( - 0.1 , 0.1 ) * model.p_da_pred[t]

model.p_rt_scenarios = Param(range(price_scenarios), model.T, initialize=real_time_price_matrix_rule)

def random_price_rule(model, t):
    return model.p_da_act[t] == model.p_da_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty) \
           + random.uniform(- model.uncertainty, model.uncertainty)

model.p_da_act_constraint = Constraint(model.T, rule=random_price_rule)


# Load Scenario Matrix and Actual Load

def random_load_matrix_rule(model, s, t):
    return model.load_scenarios[s, t] == (model.l_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty)
                                          + random.uniform(- model.uncertainty, model.uncertainty))/1000

model.load_scenarios_constraint = Constraint(range(load_scenarios), model.T, rule=random_load_matrix_rule)


def random_load_rule(model, t):
    return model.l_act[t] == model.l_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty) \
           + random.uniform(- model.uncertainty, model.uncertainty)

model.l_act_constraint = Constraint(model.T, rule=random_load_rule)


# Total Cost Scenario Matrix for All Possible Loads and Prices

def prices_calculation_rule(model, s1, s2):
    return model.prices[s1,s2] == sum(model.price_scenarios[s1,t] * (model.load_scenarios[s2,t] +
                                                                     sum(model.c[t,i] - model.efficiency_d[i] *
                                                          model.d[t,i] for i in model.i))
                                      for t in model.T)

model.price_calculations = Constraint(range(price_scenarios), range(load_scenarios), rule=prices_calculation_rule)

def cost_expectation_rule(model):
    #return model.pred_cost == sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T)
    return model.pred_cost == sum(model.prices[s1,s2] for s1 in range(price_scenarios)
                                  for s2 in range(load_scenarios))/(price_scenarios*load_scenarios) + \
           sum(model.prices_rt[s1,s2] for s1 in range(price_scenarios)
                                  for s2 in range(load_scenarios))/(price_scenarios*load_scenarios)

model.cost_expectation = Constraint(rule=cost_expectation_rule)


def cost_expectation_no_storage_rule(model):
    return model.no_storage_cost == sum(model.price_scenarios[s1,t] * model.l_pred[t]/1000 for t in model.T
                                        for s1 in range(price_scenarios))/price_scenarios + \
           sum(model.p_rt_scenarios[s1,t] * (model.l_act[t] - model.l_pred[t])/1000 for t in model.T
                                        for s1 in range(price_scenarios))/price_scenarios


model.cost_expectation_no_storage = Constraint(rule=cost_expectation_no_storage_rule)

# Real-Time Cost Scenarios

def real_time_matrix_rule(model, s, t):
    return model.load_diff_rt[s,t] == model.l_pred[t]/1000 - model.load_scenarios[s,t]

model.real_time_matrix_constraint = Constraint(range(load_scenarios), model.T, rule=real_time_matrix_rule)

def prices_rt_calculation_rule(model, s1, s2):
    return model.prices_rt[s1,s2] == sum(model.p_rt_scenarios[s1,t] * model.load_diff_rt[s2,t] for t in model.T)

model.price_rt_calculations = Constraint(range(price_scenarios), range(load_scenarios), rule=prices_rt_calculation_rule)

def total_price_calculation(model, i, j):
    return model.total_prices[i,j] == model.prices[i,j] + model.prices_rt[i,j]

model.total_prices_constraint = Constraint(range(price_scenarios), range(load_scenarios), rule=total_price_calculation)

# CVaR Constraints and Calcuation

def cvar_first_rule(model, i, j):
    return model.total_prices[i,j] - model.z - model.y[i,j] <= 0

model.cvar_first_constraint = Constraint(range(price_scenarios), range(load_scenarios), rule=cvar_first_rule)

def cvar_second_rule(model, i, j):
    return model.y[i, j] >= 0

model.cvar_second_constraint = Constraint(range(price_scenarios), range(load_scenarios), rule=cvar_second_rule)

def cvar_calculation_rule(model):
    return model.cvar == (model.z +
                          (1/(1 - model.alpha)) * 1/(price_scenarios * load_scenarios) *
                          (sum(model.y[i,j] for i in range(price_scenarios) for j in range(load_scenarios))))

model.cvar_calculation_constraint = Constraint(rule=cvar_calculation_rule)


#
# Stage-specific Cost Computations
#

def ComputeFirstStageCost_rule(model):
    return 0

model.FirstStageCost = Expression(rule=ComputeFirstStageCost_rule)

def ComputeSecondStageCost_rule(model):
    return sum(model.u_sch[t] * model.p_da_act[t] for t in model.T)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)

def ComputeThirdStageCost_rule(model):
    return sum((model.l_act[t]/1000 - model.u_sch[t]) * model.p_rt[t] for t in model.T)

model.ThirdStageCost = Expression(rule=ComputeThirdStageCost_rule)


#
# Objectives
#

def day_ahead_obj_rule(model):
    return sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T) + \
           model.beta * (model.z + \
                         (1/(1 - model.alpha)) * 1/(price_scenarios * load_scenarios) *
                         sum(model.y[i,j] for i in range(price_scenarios) for j in range(load_scenarios)))

model.day_ahead_rule = Objective(rule=day_ahead_obj_rule, sense=minimize)

