import json
import argparse
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import parametrized_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Bitaxe Hashrate Benchmark Tool')
    parser.add_argument('bitaxe_ip', nargs='?', help='IP address of the Bitaxe (e.g., 192.168.2.26 or example)')
    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

args = parse_arguments()

root_path = __file__.replace("run_analysis.py",f"/data/bitaxe_benchmark_results_{args.bitaxe_ip}.json")
with open(root_path) as data:
    jdata = json.load(data)
    df = pd.json_normalize(jdata['all_results'])

print(df)
print(df.columns)

OPT = parametrized_model([[1000,1350],[450,610]],1276)


data = df[['coreVoltage','frequency','averageTemperature','averageHashRate']].values.tolist()
for row in data:
    core,freq,temp,hashrate = row
    OPT.add_point([core,freq],hashrate,temp,core)

# filter
df = df.loc[df.coreVoltage<1350,:]
#optimise history with respect to 
min_err = 1000000
best_hist = []
for i in OPT.history:
    for j in OPT.history:
        if i != j:
    
            history = [i,j]
            OPT.reparametrize_T_eqn(history)
            OPT.reparametrize_vmin_eqn(history)
            OPT.reparametrize_t0_eqn(history)
            OPT.reparametrize_vpower(history)
            hash_rate_err= 0
            for core,freq,temp,hashrate in data:
                ypred = OPT.model([core,freq])
                errH = abs(hashrate-ypred[0])
                errT = abs(temp-ypred[1])
                hash_rate_err+=errH

            if hash_rate_err<min_err:
                min_err = hash_rate_err
                print(min_err)
                best_hist = [i,j]


OPT.reparametrize_T_eqn(best_hist)
OPT.reparametrize_vmin_eqn(best_hist)
OPT.reparametrize_t0_eqn(best_hist)
OPT.reparametrize_vpower(best_hist)



print('Vmin',OPT.vpower,OPT.k_fvmult,OPT.k_fvoffset)
print('T',OPT.tpower,OPT.k_tmult,OPT.k_toffset)
print('T0',OPT.t0)
print(OPT.get_best()[:2])

hash_rate_err= 0
for core,freq,temp,hashrate in data:
    ypred = OPT.model([core,freq])
    errH = abs(hashrate-ypred[0])
    errT = abs(temp-ypred[1])
    hash_rate_err+=errH

    print(core,freq,errH,ypred[0],errT,ypred[1])

vcore_values = np.linspace(1050, 1350, 50)  # Example range for Vcore
f_values = np.linspace(350, 650, 50)      # Example range for F

grid = [(v, f) for v in vcore_values for f in f_values]
predicted_scores = [OPT.model(x)[0] for x in grid]
vcore_grid, f_grid = np.meshgrid(vcore_values, f_values)
predicted_grid = np.array(predicted_scores).reshape(len(f_values), len(vcore_values))


# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.contourf(vcore_grid, f_grid, predicted_grid, cmap='viridis')
# plt.colorbar(label='Predicted Score')
# plt.title('Predicted Scores')
# plt.xlabel('Vcore')
# plt.ylabel('F')


# plt.subplot(1, 2, 2)
# plt.title('Real Benchmarked Data')
# plt.scatter(x=df['coreVoltage'],y=df['frequency'],c=df['averageHashRate'])
# plt.colorbar(label='Hashrate')
# plt.xlabel('Vcore')
# plt.ylabel('F')

# plt.tight_layout()
# plt.show()
plt.figure(figsize=(8, 6))
# Plot the predicted scores as a contour plot
plt.contourf(vcore_grid, f_grid, predicted_grid, cmap='viridis', alpha=0.7)
plt.colorbar(label='Predicted Score')

# Overlay the real benchmarked data as a scatter plot
plt.scatter(df['coreVoltage'], df['frequency'], c=df['averageHashRate'], cmap='viridis', edgecolor='k')
plt.colorbar(label='Hashrate')

# Titles and labels
plt.title('Overlay: Predicted Scores and Real Benchmarked Data')
plt.xlabel('Vcore')
plt.ylabel('F')

plt.tight_layout()
plt.show()
