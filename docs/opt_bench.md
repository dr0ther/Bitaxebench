# Research: Optimal benchmarking
This Program helps find optimal benchmarking settings\
weather you have done many runs already or if you want to have less iterations\
before knowing the maximum this program will analyze your benchmarking history\ 
and give you the best location to next test


## Formalising Hashrate Optimisation
the goal is to maximise hashrate, well what makes hashrate? 

### **Simple model**
this is the simpliest form which is approximatly accurate\
`Expected Hashrate = E(H) = frequency * cores`


### **Penalties**
there are penalties to this function we have temperature and voltage interactions

1. Vcore min\
if we dont have enough vcore for a frequency we cant use it\
`Vmin_penalty = Vmin-Vcore`

Penalty function I chose sigmoid as it has a gradient.\
`sigmoid = 1/(1+e^-x) = sig(x)`\
rather than a step function changing from 0-> for some value x, it is not a instantanious transition



### **Relations**
`vmin = k * freq`\
vmin relates to the frequency high frequency requires more power 

`Hashrate = E(H) * Vmin_penalty`\
`         = E(H)* sig(Vmin-Vcore)`

hashrate is dependant on 
1. difference of voltage to volatage minimum
2. expected hashrate

`T = Vcore * k + c`\
temperature is dependant on vcore

## The Hashrate function
here is the benchmarked data and fitted curve
![Example](/data/plot_ref.png)

# Results Analysis
for the experiment i am using the above function as the basis for performance comparison

I used 3 different methods to search 
*  [baseline search](https://github.com/mrv777/Bitaxe-Hashrate-Benchmark)
* hysterisis algorithom based on local history and results
* parameterized hashrate model 

Assumtions
* BM1368 

## Result (1% noise)
the perfect function is the Hashrate function i showed earlier with a tiny amount of noise
the noise was approximatly 1%
![Example](/data/convergence_analysis_pure_5.png)


|Metric | improvement from baseline |
| - | - |
|sample efficiancy improvement | 3.6X |
|convergent improvement        | 1X |

Comments\
in this case the model has reduced the time to convergence both the baseline and hysterisis algorithom


## Result (40% noise)

the noisy function is the hashrate function i showed earlier with a large amount of noise
![Example](/data/convergence_analysis_noisy_200.png)
|Metric | improvement from baseline |
| - | - |
|sample efficiancy improvement | 2X |
|convergent improvement        | 20X |

Comments\
in this case the model has outperformed the metrics by multiples in both categories