import os

myCmd = 'runef -m models -s nodedata --solve --solver=gurobi --solution-writer=pyomo.pysp.plugins.csvsolutionwriter'

os.system(myCmd)

import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('ef.csv')

prices = results.iloc[171:1170,4]

var = results.iloc[119,4]
cvar = results.iloc[120,4]
pred_cost = results.iloc[170,4]

no_storage_cost = results.iloc[145,4]

profit = no_storage_cost - prices
profit_cvar = no_storage_cost - cvar
profit_var = no_storage_cost - var
profit_pred_cost = no_storage_cost - pred_cost

"""
num_bins = 40
n, bins, patches = plt.hist(prices, num_bins, facecolor='blue', alpha=0.4)
plt.axvline(cvar, color='k', linestyle='dashed', linewidth=1)
plt.text(cvar-20, 55, 'CVaR', fontsize=10, bbox = {'facecolor' : 'white'}) #: {:.2f}.format(cvar)
plt.axvline(var, color='k', linestyle='dashed', linewidth=1)
plt.text(var-180, 55, 'VaR', fontsize=10, bbox = {'facecolor' : 'white'})
plt.axvline(pred_cost, color='r', linestyle='dashed', linewidth=1)
plt.text(pred_cost-180, 5, 'Predicted \n Cost', fontsize=10, color='red')
plt.xlabel('Day-Ahead Market Generated Scenarios (£)')
plt.title('200 Cost Scenarios and 10 Load Scenarios, Uncertainty = 15%, Alpha=0.95, Beta=0.1', fontsize=10)
plt.ylabel('Frequency')
plt.savefig('Images/Histogram_alpha_95_u_15_b_0p1.pdf')
plt.show()
"""


num_bins = 40
n, bins, patches = plt.hist(profit, num_bins, facecolor='blue', alpha=0.4)
plt.axvline(profit_cvar, color='k', linestyle='dashed', linewidth=1)
plt.text(profit_cvar, 30, 'CVaR', fontsize=10, bbox = {'facecolor' : 'white'}) #: {:.2f}.format(cvar)
plt.axvline(profit_var, color='k', linestyle='dashed', linewidth=1)
plt.text(profit_var, 40, 'VaR', fontsize=10, bbox = {'facecolor' : 'white'})
plt.axvline(profit_pred_cost, color='r', linestyle='dashed', linewidth=1)
plt.text(profit_pred_cost, 5, 'Predicted \n Profit', fontsize=10, color='red')
plt.xlabel('Day-Ahead Market Generated Scenarios (£)')
#plt.title('1000 Cost Scenarios, Uncertainty = 5%, Alpha=0.95, Beta=0.1', fontsize=10)
plt.ylabel('Frequency')
#plt.savefig('Images/Histogram_alpha_95_u_15_b_0p1.pdf')
plt.show()

print(profit_cvar)

load = results.iloc[121:145,4:5].reset_index(drop=True)

print(load)