import json
import argparse
import sys
import pandas as pd # TODO: REMOVE ME make optional

from model import parametrized_model,parametrized_model2

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
    try:
        df = pd.json_normalize(jdata['all_results'])
    except:
        try:
            df = pd.json_normalize(jdata)
        except:
            print("cant read file")
            raise ValueError("json not in expected form")

model = parametrized_model2(1276)

data = df[['coreVoltage','frequency','averageTemperature','averageHashRate','efficiencyJTH']].values.tolist()
for row in data:
    core,freq,temp,hashrate,eff = row
    model.add_point([core,freq],hashrate,temp,eff)

# filter
df = df.loc[df.coreVoltage<1350,:]

model.build(model.history)
print(model.is_trained)


print("\nErrors")
print(f'Hashrate err ={model.calc_err(model.history)}')
#print(f'Temp err     ={errT/len(data)}')

# Calulate best postition from model
best_loc = model.maximise_hashrate_eqn([[1050,1250],[400,625]])
print(best_loc)
print("\nThe Best Positions")
print("Vcore Freq Hashrate")
print("    Type   V    F   H")
print("Benchmark ",int(model.get_history_best()[0][0]),int(model.get_history_best()[0][1]),int(model.get_history_best()[1]))
print("    Model ",best_loc[0],best_loc[1],int(model.evaluate(best_loc)))

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
    vcore_values = np.linspace(1050, 1450, 100)  # Example range for Vcore
    f_values = np.linspace(350, 700, 100)      # Example range for F

    grid = [(v, f) for v in vcore_values for f in f_values]
    predicted_scores = [model.evaluate(x) for x in grid]
    vcore_grid, f_grid = np.meshgrid(vcore_values, f_values)
    predicted_grid = np.array(predicted_scores).reshape(len(f_values),len(vcore_values)).T

    # Normalize to the combined range of predicted and real data
    min_val = min(predicted_grid.min(), df['averageHashRate'].min())
    max_val = max(predicted_grid.max(), df['averageHashRate'].max())
    norm = Normalize(vmin=min_val, vmax=max_val)

    # Plot the predicted scores as a contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(vcore_grid, f_grid, predicted_grid, cmap='viridis', norm=norm, alpha=0.6)

    # Overlay the real benchmarked data as a scatter plot
    scatter = plt.scatter(
        df['coreVoltage'],
        df['frequency'],
        c=df['averageHashRate'],
        cmap='viridis',
        norm=norm,
        edgecolor='k',
    )

    plt.plot(best_loc[0], best_loc[1], marker='*', color='red', markersize=10, label='Best Position')


    # Add a single shared color bar for both plots
    cbar = plt.colorbar(contour)
    cbar.set_label('Hashrate')
    plt.title('Hashrate Model (Translucent) and Benchmarked Data')
    plt.xlabel('Vcore (mV)')
    plt.ylabel('Frequency (MHz)')
    plt.tight_layout()

    # Save the plot to a file
    output_file = f"plot_{args.bitaxe_ip}.png"  # Replace with your desired filename
    plt.savefig(output_file)
    print("\nGraph")
    print(f"Plot saved to {output_file}")