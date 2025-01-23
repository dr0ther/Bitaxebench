import json
import argparse
import sys
import pandas as pd # TODO: REMOVE ME make optional

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


OPT = parametrized_model([[1000,1350],[450,610]],1276)


data = df[['coreVoltage','frequency','averageTemperature','averageHashRate']].values.tolist()
for row in data:
    core,freq,temp,hashrate = row
    OPT.add_point([core,freq],hashrate,temp,core)

# filter
df = df.loc[df.coreVoltage<1350,:]



#optimise history with respect to Hashrate error
best_hist = []
min_err = 1000000
best_hist = OPT.history
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
                best_hist = [i,j]

OPT.reparametrize_T_eqn(best_hist)
OPT.reparametrize_vmin_eqn(best_hist)
OPT.reparametrize_t0_eqn(best_hist)
OPT.reparametrize_vpower(best_hist)


print("\nLearned voltage and temperature penalty parameters")
print('Vmin = p * (F * k + c )')
print(f"p={OPT.vpower:.5} k={OPT.k_fvmult:.5} c={OPT.k_fvoffset:.5}")
print('\np * (T0 - (V * k + c))')
print(f"p={OPT.tpower:.5} k={OPT.k_tmult:.5} c={OPT.k_fvoffset:.5}")
print(f'\nT0={float(OPT.t0):.5}')

hash_rate_err= 0
for core,freq,temp,hashrate in data:
    ypred = OPT.model([core,freq])
    errH = abs(hashrate-ypred[0])
    errT = abs(temp-ypred[1])
    hash_rate_err+=errH

print("\nErrors")
print(f'Hashrate err ={errH/len(data)}')
print(f'Temp err     ={errT/len(data)}')

# Calulate best postition from model
best_p = OPT.maximise_eqn()

print("\nThe Best Positions")
print("Vcore Freq Hashrate")
print("    Type   V    F   H")
print("Benchmark ",int(OPT.get_best()[0][0]),int(OPT.get_best()[0][1]),int(OPT.get_best()[1]))
print("    Model ",best_p[0],best_p[1],int(OPT.model(best_p)[0]))

to_graph = False
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    to_graph = True

except ImportError:
    print("missing libraries ignore if you dont want graphs")
    print("pip install pandas matplotlib numpy")




if to_graph:
    vcore_values = np.linspace(1050, 1350, 50)  # Example range for Vcore
    f_values = np.linspace(350, 650, 50)      # Example range for F

    grid = [(v, f) for v in vcore_values for f in f_values]
    predicted_scores = [OPT.model(x)[0] for x in grid]
    vcore_grid, f_grid = np.meshgrid(vcore_values, f_values)
    predicted_grid = np.array(predicted_scores).reshape(len(f_values), len(vcore_values))

    # Normalize to the combined range of predicted and real data
    min_val = min(predicted_grid.min(), df['averageHashRate'].min())
    max_val = max(predicted_grid.max(), df['averageHashRate'].max())
    norm = Normalize(vmin=min_val, vmax=max_val)



    # Make graph
    plt.figure(figsize=(8, 6))

    # Plot the predicted scores as a contour plot
    contour = plt.contourf(vcore_grid, f_grid, predicted_grid, cmap='viridis', norm=norm, alpha=0.7)

    # Overlay the real benchmarked data as a scatter plot
    scatter = plt.scatter(
        df['coreVoltage'],
        df['frequency'],
        c=df['averageHashRate'],
        cmap='viridis',
        norm=norm,
        edgecolor='k',
    )

    # Add a single shared color bar for both plots
    cbar = plt.colorbar(contour)
    cbar.set_label('Hashrate')

    # Titles and labels
    plt.title('Hashrate Model (Translucent) and Benchmarked Data')
    plt.xlabel('Vcore (mV)')
    plt.ylabel('Frequency (MHz)')

    plt.tight_layout()

    # Save the plot to a file
    output_file = "plot.png"  # Replace with your desired filename
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")