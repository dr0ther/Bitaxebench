# Bitaxe Hashrate Benchmark

A Python-based benchmarking tool for optimizing Bitaxe mining performance by testing different voltage and frequency combinations while monitoring hashrate, temperature, and power efficiency.

## Credits
- `Whitey Cookie`
- `mv777`
- `shufps`


## Features

- Automated benchmarking of different voltage/frequency combinations
- Temperature monitoring and safety cutoffs
- Power efficiency calculations (J/TH)
- Automatic saving of benchmark results
- Graceful shutdown with best settings retention
- Docker support for easy deployment
- Modeling of Hashrate equation

## Formalising Hashrate Optimisation
the goal is to maximise hashrate, well what makes hashrate? 

### **Simple model**
this is the simpliest form which is approximatly accurate\
`Expected Hashrate = E(H) = frequency * cores`


### **Penalties**
there are penalties to this function we have temperature and voltage interactions

1. High temps\
if we have a high temperature we get less hashrate\
`Temp_penalty = T0-tempurature` 


2. Vcore min\
if we dont have enough vcore for a frequency we cant use it\
`Vmin_penalty = Vmin-Vcore`

Penalty function I chose sigmoid as it has a gradient.\
`sigmoid = 1/(1+e^-x) = sig(x)`\
rather than a step function changing from 0-> for some value x, it is not a instantanious transition



### **Relations**
`vmin = k * freq`\
vmin relates to the frequency high frequency requires more power 

`T = k * vcore`\
temperature increases when more power is transmitted to the chip

`T0` T0 is the overheat temperature in my model it is static


this results in a complicated equation\
`Hashrate = E(H) * Temp_penalty * Vmin_penalty`\
`         = E(H) * sig(T0-T) * sig(Vmin-Vcore)`

hashrate is dependant on 
1. temperature difference from current to overheat
2. minimum required voltage for a frequncy
3. difference of voltage to volatage minimum
4. expected hashrate

`T = Vcore * k + c`\
temperature is dependant on vcore


### **Code**
plotly pandas matplotlib requried for graphs\
the code 
```bash 
python run_analysis.py filename
``` 
parametrizes the above function with your benchmark data  **it supports old data** you dont need to rerun you benchmarks


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
python run_analysis.py example
``` 
will display the following 

![Example](/data/plot_example.png)

this shows the learned model and the best parts


### Model limitations
the model fails in various aspects as there are many assumptions that are baked in
for instance t0 > 75 




## Prerequisites

- Python 3.11 or higher
- Access to a Bitaxe miner on your network
- Docker (optional, for containerized deployment)

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/adammwest/Bitaxebench/
cd Bitaxebench
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t bitaxe-benchmark .
```

## Usage

### Standard Usage

Run the benchmark tool by providing your Bitaxe's IP address:

```bash
python bitaxe_hashrate_benchmark.py <bitaxe_ip>
```

Optional parameters:
- `-v, --voltage`: Initial voltage in mV (default: 1150)
- `-f, --frequency`: Initial frequency in MHz (default: 500)

Example:
```bash
python bitaxe_hashrate_benchmark.py 192.168.2.26 -v 1200 -f 550
```

### Docker Usage

Run the container with your Bitaxe's IP address:

```bash
docker run --rm bitaxe-benchmark <bitaxe_ip> [options]
```

Example:
```bash
docker run --rm bitaxe-benchmark 192.168.2.26 -v 1200 -f 550
```

## Configuration

The script includes several configurable parameters:

- Maximum chip temperature: 66°C
- Maximum VR temperature: 90°C
- Maximum allowed voltage: 1300mV
- Maximum allowed frequency: 650MHz
- Benchmark duration: 2.5 minutes
- Sample interval: 30 seconds

## Output

The benchmark results are saved to `bitaxe_benchmark_results.json`, containing:
- Complete test results for all combinations
- Top 5 performing configurations ranked by hashrate
- For each configuration:
  - Average hashrate
  - Temperature readings
  - Power efficiency metrics (J/TH)
  - Voltage/frequency combinations tested

## Safety Features

- Automatic temperature monitoring with safety cutoff (66°C chip temp)
- Voltage regulator (VR) temperature monitoring with safety cutoff (90°C)
- Graceful shutdown on interruption (Ctrl+C)
- Automatic reset to best performing settings after benchmarking
- Input validation for safe voltage and frequency ranges
- Hashrate validation to ensure stability

## Benchmarking Process

The tool follows this process:
1. Starts with user-specified or default voltage/frequency
2. always find stable temp before starting benchmark sample
3. Uses PSO algorithom with a Cost function to optimise a numeric goal e.g hashrate or efficiency
4. Records and ranks all successful configurations
5. Automatically applies the best performing stable settings

## Todo 
* add Shufps test stratum server for the same jobs and allowing lower pool diff reducing benchmarking time and hashrate variance


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.





## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Please use this tool responsibly. Overclocking and voltage modifications can potentially damage your hardware if not done carefully. Always ensure proper cooling and monitor your device during benchmarking.
