# test
# mrv iteration
# hysterisis
# param model
import pandas as pd
import json
from model import parametrized_model2
import random
import plotly.express as px
import matplotlib.pyplot as plt

bitaxe_ip = "192.168.1.139_ref"

root_path = __file__.replace("test_bench.py",f"/data/bitaxe_benchmark_results_{bitaxe_ip}.json")
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
        

ref = parametrized_model2(1276)

data = df[['coreVoltage','frequency','averageTemperature','averageHashRate','efficiencyJTH']].values.tolist()
for row in data:
    core,freq,temp,hashrate,eff = row
    ref.add_point([core,freq],hashrate,temp,eff)

ref.build(ref.history)
bounds = [[1100,1275],[400,625]]
goal_p = ref.maximise_hashrate_eqn(bounds)
ref_max = ref.evaluate(goal_p)
print('goal',ref_max,goal_p)


def dist_from_best(p):
    dist_vec = [abs(p[0]-goal_p[0]),abs(p[1]-goal_p[1])]
    return dist_vec,sum(dist_vec)


def noisy_eval(p):
    noise_p = 50
    return max(ref.evaluate(p)+(noise_p*2*random.random()-noise_p),0)
#assess convergence from 1100,400 starting point

hysterisis_update = 25
def bench_model():
    all_history = []
    model = parametrized_model2(1276)
    results =[]
    initial_probes = [[1100,400]]
    best = 0
    itr = 0 
    history = []
    
    p = initial_probes[0]
    results,history = run_hyserisis(p,4)
    for h in history:
        model.add_point(h[0],h[1],1,1)
        itr+=1
    
    #print(model.history)
    model.build(model.history)
    best_model = model

    p = initial_probes[0]
    for i in range(53):
        
        model.build_partial(model.history)
        model_err = model.calc_err(model.history)
        best_model_err = best_model.calc_err(best_model.history)
        
        #print('new model',model_err,best_model_err)
        if model_err<best_model_err:
            best_model_err = model_err
            best_model = model
                
        hint = i%2==0 and i > 12
        suggestion = best_model.maximise_hashrate_eqn(bounds,hint)
        new_v = suggestion[0]
        new_f = suggestion[1]
        p = [int(new_v),int(new_f)]

        score = noisy_eval(p)
        itr+=1
        if ref.evaluate(p)>best:
            best=ref.evaluate(p)

        print(i,p,dist_from_best(p),int(ref.evaluate(p)),int(best_model.calc_err(best_model.history)),int(best_model.evaluate(p)))

        model.add_point(p,score,1,1)
        best_model.add_point(p,score,1,1)
        convergence_prcnt = best/ref_max
        results.append([itr,best,convergence_prcnt,ref.evaluate(p),score,p[0],p[1]])
        all_history.append([p,score])
            
        if convergence_prcnt > 0.99: 
            results.append([itr+1,best,1,ref.evaluate(p),score,p[0],p[1]])
            break
        
        if i == 49:
            results.append([itr+1,best,0,ref.evaluate(p),score,p[0],p[1]])

    

    return results


def bench_basline():
    initial_probes = [[1100,400]]
    results = []
    best = 0
    itr = 0 
    p = initial_probes[0]
    for i in range(65):
    
        
        score = noisy_eval(p)

        if ref.evaluate(p)>best:
            best=ref.evaluate(p)


        convergence_prcnt = best/ref_max
        results.append([itr,best,convergence_prcnt,ref.evaluate(p),score,p[0],p[1]])

        #update
        ok = (score >= ref.eval_Expected_H(p) * 0.92)
        if ok:
             p = [p[0],p[1]+hysterisis_update] 
        
        if not ok:
            p = [p[0]+hysterisis_update,p[1]-hysterisis_update]

        # exit conditions
        if convergence_prcnt > 0.99: 
            results.append([itr+1,best,1,ref.evaluate(p),score,p[0],p[1]])
            break
        if p[0] > bounds[0][1] or p[1] > bounds[1][1]: 
            results.append([itr+1,best,0,ref.evaluate(p),score,p[0],p[1]])
            break
        itr+=1
    return results




def run_hyserisis(x0,n):
    results = []
    history = []
    p = x0
    itr = 0
    best = 0
    for i in range(n):
        score = noisy_eval(p)

        if ref.evaluate(p)>best:
            best=ref.evaluate(p)

        convergence_prcnt = best/ref_max
        results.append([itr,best,convergence_prcnt,ref.evaluate(p),score,p[0],p[1]])
        history.append([p,score])
        # update
        if i == 0: p = [p[0],p[1]+hysterisis_update] 
        

        if len(history)>=2:
            is_v_inc = history[-1][0][0]>history[-2][0][0]
            is_f_inc = history[-1][0][1]>history[-2][0][1]
            is_h_inc = history[-1][1]>history[-2][1]
            if is_f_inc:
                if is_h_inc:
                    p = [p[0],p[1]+hysterisis_update] 
                else:
                    p = [p[0]+hysterisis_update,p[1]] 

            if is_v_inc:
                if is_h_inc:
                    p = [p[0],p[1]+hysterisis_update] 
                else:
                    p = [p[0]+hysterisis_update,p[1]] 
            

        # exit conditions
        if convergence_prcnt > 0.99: 
            results.append([itr+1,best,1,ref.evaluate(p),score,p[0],p[1]])
            break
        if p[0] > bounds[0][1] or p[1] > bounds[1][1]: 
            results.append([itr+1,best,0,ref.evaluate(p),score,p[0],p[1]])
            break

        itr+=1
    return results,history


def bench_adam():
    initial_probes = [[1100,400]]
    results,history = run_hyserisis(initial_probes[0],65)
    return results

        
data = []
for i in range(250):
 for r in bench_model():
    data.append([i,'model',*r])

for i in range(250):
 for r in bench_basline():
    data.append([i,'baseline',*r])

for i in range(250):
 for r in bench_adam():
    data.append([i,'hysteresis',*r])


df = pd.DataFrame(data,columns="exp_n,bench_type,itr_n,best_val,converge%,ref_val,score_val,v,f".split(","))
df.exp_n = df.exp_n.astype(str)
df.loc[:,'expn_type'] = df.exp_n+"_"+df.bench_type
df.loc[:,'itr%'] = df.itr_n/55
df.loc[:,'convergence'] = df["converge%"]/df['itr_n']
px.scatter(df,x='itr_n',y='v',color='bench_type').show()
px.scatter(df,x='itr_n',y='f',color='bench_type').show()
px.scatter(df,x='itr_n',y='converge%',color='bench_type').show()

last_itr_df = (
    df.groupby(["exp_n", "bench_type"], as_index=False) 
    .last() 
)
convergence_percentage = last_itr_df.groupby("bench_type")['converge%'].describe()
convergence_time = last_itr_df.groupby("bench_type")['itr_n'].describe()

# Define the desired order of bench types
desired_order = ['model', 'hysteresis', 'baseline']
convergence_percentage = convergence_percentage.reindex(desired_order)
convergence_time = convergence_time.reindex(desired_order)

percentage_means = convergence_percentage["mean"]
percentage_stds = convergence_percentage["std"]
time_means = convergence_time["mean"]
time_stds = convergence_time["std"]
bench_types = convergence_percentage.index

# Plot settings
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# Convergence Percentage Plot
axes[0].bar(bench_types, percentage_means, yerr=percentage_stds, color=['blue', 'green', 'red'], alpha=0.7, capsize=5)
axes[0].set_title("Convergence % by Bench Type (Higher better)")
axes[0].set_xlabel("Bench Type")
axes[0].set_ylabel("Convergence %")
axes[0].set_ylim(0, 1.2)  # Adjust as needed
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Convergence Time Plot
axes[1].bar(bench_types, time_means, yerr=time_stds, color=['blue', 'green', 'red'], alpha=0.7, capsize=5)
axes[1].set_title("Convergence Time by Bench Type (Lower better)")
axes[1].set_xlabel("Bench Type")
axes[1].set_ylabel("Convergence Time (Avg)")
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("convergence_analysis.png", dpi=300)  # Save as a high-resolution PNG file
plt.show()
