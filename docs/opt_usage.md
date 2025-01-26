# Usage

you need pandas and matplotlib for the graphs, the model is pure python


### **Code**
plotly pandas matplotlib requried for graphs\
the code 
```bash 
python run_analysis.py filename
``` 
parametrizes the above function with your benchmark data  it supports old data you dont need to rerun you benchmarks


### Code examples
1. analysis on the refrence benchmark
```bash 
python run_analysis.py ref
``` 

2. analysis on the example benchmark

```bash 
python run_analysis.py example
``` 





### Interpretting output
the program will output somthing like
```
Learned voltage and temperature penalty parameters
Voltage penalty p * (F * k + c )
p=0.36 k=0.76289 c=797.56

Temp penalty p * (T0 - (V * k + c))
p=-0.13258 k=0.093341 c=797.56

T0=75.0

Errors
Hashrate err =0.05073310330461451
Temp err     =9.568037816170017

The Best Positions
Vcore Freq Hashrate
    Type   V    F   H
Benchmark  1203 508 649
    Model  1270 600 671
Plot saved to plot.png
```


1. The Errors 
this section gives the model errors to the hashrate and temperature equations

2. The Best Positions
this is the most intresting section for the user 
```
    Model  1270 600 671
```
this says the model thinks vcore=1270 and freq=600 will result in hashrate=671\
it is estimating the maximum it may be good to try that combination


### **Graph**
```bash
python run_analysis.py <bitaxe_ip>
``` 


### Model limitations
the model fails in various aspects as there are many assumptions that are baked in
for instance t0 > 75 