
from pyomo.environ import *
import random
import numpy as np

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

model.l_act = Param(model.T)

model.p_da_pred = Param(model.T, within=PositiveReals)

model.gradient = Param(within=PositiveReals)

model.alpha = Param(within=PositiveReals)

model.beta = Param(within=PositiveReals)

model.uncertainty = Param(within=PositiveReals)

price_scenarios = 1000

load_scenarios = 1

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


# Price Variables

model.p_pm_pred = Var(model.T)

model.p_da_act = Var(model.T)

model.prices = Var(range(price_scenarios), range(load_scenarios))

model.prices_rt = Var(range(price_scenarios), range(load_scenarios))

model.pred_cost = Var()

model.prices_no_storage = Var(range(price_scenarios), range(load_scenarios))

model.no_storage_cost = Var()


# Load Variables

model.load_scenarios = Var(range(load_scenarios), model.T)

model.load_scenarios_no_storage = Var(range(load_scenarios), model.T)

model.l_act = Var(model.T)

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

def real_time_price_rule(model, t):
    return model.p_da_pred[t] * random.uniform(0.5, 1.5) + \
           random.uniform( - 0.1 * model.p_da_pred[t], 0.1 * model.p_da_pred[t])

model.p_rt = Param(model.T, initialize=real_time_price_rule)

def real_time_price_matrix_rule(model, s, t):
    return model.p_da_pred[t] * random.uniform(0.5, 1.5) + \
           random.uniform( - 0.1 * model.p_da_pred[t], 0.1 * model.p_da_pred[t])

model.p_rt_scenarios = Param(range(price_scenarios), model.T, initialize=real_time_price_matrix_rule)

def random_price_rule(model, t):
    return model.p_da_act[t] == model.p_da_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty) \
           + random.uniform(- model.uncertainty, model.uncertainty)

model.p_da_act_constraint = Constraint(model.T, rule=random_price_rule)


# Load Scenario Matrix and Actual Load

def random_load_matrix_rule(model, s, t):
    return model.load_scenarios[s, t] == (model.l_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty)
                                          + random.uniform(- model.uncertainty, model.uncertainty))/1000 + \
           sum(model.c[t, i] - model.efficiency_d[i] * model.d[t, i] for i in model.i)

model.load_scenarios_constraint = Constraint(range(load_scenarios), model.T, rule=random_load_matrix_rule)


def random_load_rule(model, t):
    return model.l_act[t] == model.l_pred[t] * random.uniform(1 - model.uncertainty, 1 + model.uncertainty) \
           + random.uniform(- model.uncertainty, model.uncertainty)

model.l_act_constraint = Constraint(model.T, rule=random_load_rule)


# Total Cost Scenario Matrix for All Possible Loads and Prices

def prices_calculation_rule(model, s1, s2):
    return model.prices[s1,s2] == sum(model.price_scenarios[s1,t] * model.load_scenarios[s2,t] for t in model.T)

model.price_calculations = Constraint(range(price_scenarios), range(load_scenarios), rule=prices_calculation_rule)

def cost_expectation_rule(model):
    #return model.pred_cost == sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T)
    return model.pred_cost == sum(model.prices[s1,s2] for s1 in range(price_scenarios)
                                  for s2 in range(load_scenarios))/(price_scenarios*load_scenarios)

model.cost_expectation = Constraint(rule=cost_expectation_rule)


def cost_expectation_no_storage_rule(model):
    return model.no_storage_cost == sum(model.price_scenarios[s1,t] * model.l_pred[t]/1000 for t in model.T
                                        for s1 in range(price_scenarios))/price_scenarios

model.cost_expectation_no_storage = Constraint(rule=cost_expectation_no_storage_rule)

# Real-Time Cost Scenarios

def real_time_matrix_rule(model, s, t):
    return model.load_diff_rt[s,t] == model.load_scenarios[s,t] - model.l_act[t]

model.real_time_matrix_constraint = Constraint(range(load_scenarios), model.T)

def prices_rt_calculation_rule(model, s1, s2):
    return model.prices_rt[s1,s2] == sum(model.p_rt_scenarios[s1,t] * model.load_diff_rt[s2,t] for t in model.T)

model.price_rt_calculations = Constraint(range(price_scenarios), range(load_scenarios), rule=prices_calculation_rule)


# CVaR Constraints and Calcuation

def cvar_first_rule(model, i, j):
    return model.prices[i,j] - model.z - model.y[i,j] <= 0

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
    return sum(model.u_sch[t] * model.p_da_act[t] for t in model.T) + \
           sum((model.l_act[t]/1000 - model.u_sch[t]) * model.p_rt[t] for t in model.T)

model.SecondStageCost = Expression(rule=ComputeSecondStageCost_rule)


#
# Objectives
#

def day_ahead_obj_rule(model):
    return sum(model.u_sch[t] * model.p_da_pred[t] for t in model.T) + \
           model.beta * (model.z + \
                         (1/(1 - model.alpha)) * 1/(price_scenarios * load_scenarios) *
                         sum(model.y[i,j] for i in range(price_scenarios) for j in range(load_scenarios)))

model.day_ahead_rule = Objective(rule=day_ahead_obj_rule, sense=minimize)


"""
no_storage = 0

p_rt_pred = [55.138609,
60.941038,
58.140177,
43.713802,
58.184362,
71.822004,
63.774196,
65.221001,
66.161911,
86.264506,
69.463536,
63.759761,
63.517166,
58.383306,
32.472592,
56.892780,
69.687804,
73.508857,
68.715289,
69.340769,
54.279267,
32.369939,
72.153706,
38.294100]

l_act = [41110.57068089, 37262.07534591, 35530.42694912, 33613.90907861,
 33614.9218241,  35880.0624191,  46758.42342253, 57339.9140661,
 58839.32771865, 57283.8712273,  54769.0068911,  54034.56813284,
 55757.6658119,  53919.67918116, 54983.20504804, 61039.55217808,
 76539.19466524, 90168.43503743, 90658.54724467, 83425.9052774,
 76780.84903932, 69186.06323457, 57632.92706161, 42895.4987654 ]

u_act3 = p_rt_pt_pred = np.array([])

socj = np.array([])
for i in range(no_storage + 1):
    socj = np.append(socj, 0.4 * 1)

charge_rt = [[] for i in range(no_storage + 1)]
discharge_rt = [[] for i in range(no_storage + 1)]
soc_rt = [[] for i in range(no_storage + 1)]

u_act3_pm = p_rt_pm_pred = p_rt_pm_act = np.array([])

socj_pm = np.array([])
for i in range(no_storage + 1):
    socj_pm = np.append(socj_pm, 0.4 * 1)

charge_rt_pm = [[] for i in range(no_storage + 1)]
discharge_rt_pm = [[] for i in range(no_storage + 1)]
soc_rt_pm = [[] for i in range(no_storage + 1)]

for i in range(no_storage + 1):
    soc_rt[i] = np.append(soc_rt[i], [socj[i]])
    soc_rt_pm[i] = np.append(soc_rt_pm[i], [socj_pm[i]])

l = 0

while l <= 23:

    
    Strategy 3 (MPC Optimisation)
    

    if l == 24:
        j = 0
    else:
        j = l

    model = ConcreteModel(name="RealTime")
    model.T = RangeSet(0, 23)
    model.i = RangeSet(0, no_storage)

    # INDICES

    k = np.array([])  # future time periods
    for t in model.T:
        m = j + 1 + t
        if m > 23:
            m = m - 24
        k = np.append(k, m)
    k = k.astype(int)
    model.n = Var(model.T)
    model.m = Var(model.T)


    # VARIABLES

    def power_bounds_rule(model, t, i):
        return (0, 2 * 1)


    def capacity_bounds_rule(model, t, i):
        return (0.4 * 1, 1 * 1)


    model.c = Var(model.T, model.i, bounds=power_bounds_rule)
    model.d = Var(model.T, model.i, bounds=power_bounds_rule)
    model.x = Var(model.T, model.i, bounds=capacity_bounds_rule)
    model.u_act = Var(model.T)
    model.u_pred = Var(model.T)
    model.p_pm_pred = Var(model.T)


    # FUNCTIONS

    def soc_rule_current_period(model, i):
        return model.x[j, i] == socj[i] + 0.95 * model.c[j, i] - model.d[j, i]


    def soc_rule_current_period_pm(model, i):
        return model.x[j, i] == socj_pm[i] + 0.95 * model.c[j, i] - model.d[j, i]


    def soc_rule(model, t, i):
        if t == 0:
            s = 23
        else:
            s = t - 1
        return model.x[k[t], i] == model.x[k[s], i] + 0.95 * model.c[k[t], i] - model.d[k[t], i]


    def power_import_current_period(model):
        return model.u_act[j] == (l_act[j] / 1000) + sum(model.c[j, i] - 0.85 * model.d[j, i] for i in model.i)


    def power_import_predicted_periods(model, t):
        return model.u_pred[k[t]] == (model.l_pred[k[t]] / 1000) + sum(
            model.c[k[t], i] - 0.85 * model.d[k[t], i] for i in model.i)


    def absolute_val_max(model, t):
        return model.u_act[t] - model.u_sch[t] <= model.n[t]


    def pred_absolute_val_max(model, t):
        return model.u_pred[t] - model.u_sch[t] <= model.m[t]


    def strat3_obj_rule(model):
        return ((model.n[j] * p_rt_pred[j]) + sum(model.m[k[t]] * p_rt_pred[k[t]] for t in range(23)))


    def p_pm_current_rule(model):
        return model.p_pm_pred[j] == p_rt_pred[j] + model.gradient * (model.n[j])


    def p_pm_rule(model, t):
        return model.p_pm_pred[k[t]] == p_rt_pred[k[t]] + model.gradient * (model.m[k[t]])


    def real_time_obj_rule_pm(model):
        return ((model.n[j] * model.p_pm_pred[j]) + sum(model.m[k[t]] * model.p_pm_pred[k[t]] for t in range(23)))


    # CONSTRAINTS

    model.soc_current_period_constraint = Constraint(model.i, rule=soc_rule_current_period)
    model.soc_current_period_pm_constraint = Constraint(model.i, rule=soc_rule_current_period_pm)
    model.soc_constraint = Constraint(range(23), model.i, rule=soc_rule)
    model.absolute_val_max_constraint = Constraint(model.T, rule=absolute_val_max)
    model.pred_absolute_val_max_constraint = Constraint(model.T, rule=pred_absolute_val_max)
    model.power_predicted_constraint = Constraint(range(23), rule=power_import_predicted_periods)
    model.power_current_constraint = Constraint(rule=power_import_current_period)
    model.p_pm_constraint = Constraint(range(23), rule=p_pm_rule)
    model.p_pm_current_constraint = Constraint(rule=p_pm_current_rule)

    # OBJECTIVE

    model.real_time_objective_pm = Objective(rule=real_time_obj_rule_pm)
    model.real_time_strat_3_objective = Objective(rule=strat3_obj_rule)

    # STRATEGY 3 SOLVE

    model.real_time_objective_pm.deactivate()
    model.soc_current_period_pm_constraint.deactivate()

    solver_result = solver.solve(model)
    for i in model.i:
        charge_rt[i] = np.append(charge_rt[i], [model.c[j, i].value])
        discharge_rt[i] = np.append(discharge_rt[i], [model.d[j, i].value])
        soc_rt[i] = np.append(soc_rt[i], [model.x[j, i].value])
        socj[i] = model.x[j, i].value

    u_act3 = np.append(u_act3, model.u_act[j].value)
    p_rt_pt_pred = np.append(p_rt_pt_pred, model.p_pm_pred[j].value)

    # PRICE MAKER SOLVE

    model.real_time_strat_3_objective.deactivate()
    model.real_time_objective_pm.activate()
    model.soc_current_period_constraint.deactivate()
    model.soc_current_period_pm_constraint.activate()

    solver_result = solver.solve(model)
    for i in model.i:
        charge_rt_pm[i] = np.append(charge_rt_pm[i], [model.c[j, i].value])
        discharge_rt_pm[i] = np.append(discharge_rt_pm[i], [model.d[j, i].value])
        soc_rt_pm[i] = np.append(soc_rt_pm[i], [model.x[j, i].value])
        socj_pm[i] = model.x[j, i].value

    p_rt_pm_pred = np.append(p_rt_pm_pred, model.p_pm_pred[j].value)
    p_rt_pm_act = np.append(p_rt_pm_act, p_rt_act[j] + gradient_rt *
                            (l_act[j] / 1000 + sum(model.c[j, i].value - 0.85 * model.d[j, i].value for i in model.i)))

    u_act3_pm = np.append(u_act3_pm, model.u_act[j].value)

    l = l + 1

for i in range(no_storage + 1):
    soc_rt[i] = np.delete(soc_rt[i], -1)
    soc_rt_pm[i] = np.delete(soc_rt_pm[i], -1)

charge_total_rt = sum(charge_rt[i] for i in range(no_storage + 1))
discharge_total_rt = sum(discharge_rt[i] for i in range(no_storage + 1))
soc_total_rt = sum(soc_rt[i] for i in range(no_storage + 1))

"""