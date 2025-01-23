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
- Particle swarm optimisation
- Easy to implemnt cost functions
- Stable temps before sampling
- Independant hashrate samples only



## Optimiser
https://en.wikipedia.org/wiki/Particle_swarm_optimization
the optimiser is a global optimiser that has its inital postions spread out evenly, 
so should converge to the global minimum given we have smooth convex space to search

# Cost function
what the optimser tends towards
- `efficiancy` 
- `hashrate`
- `hashrate_expected` ratio of hashrate/expected hashrate
- `hashrate_temp` optimise temp to control temp and hashrate
- `hashrate_efficiancy` optimise hashrate to control temp and efficiancy
- `efficiancy_temp` optimise temp to control temp and efficiancy

you can add your own for any combination of 
[`efficiancy`, `temp`, `hashrate`]

# Formalising Hashrate Optimisation
https://www.rohm.com/electronics-basics/transistors/tr_what7

3 data points are needed to create the estimate of the curve

basicially you have various points



what does in entail?
at its core there is `Hashrate = frequency*C`, where C is a constant

so we can just add more frequncy rightttt?
well for more frequency we need more power, that is what vcore is for.
so to achieve a specific F there is a minimum amount of power the chip needs
`Vmin = F * c`

so we have the constraint
`Vmin < Vcore` where Vmin = ?

then even if the chip has enough power, it starts gettting more hot
when the chip is too hot it loses nonces, or hashrate.
so we have a heat penalty 
`Tp = (t-T)/C`

heat is a function of power and power relates to Voltage
`W = V * A`
so 
`Vcore = C * T` Vcore relates to tempaerature

putting it all together

`T = Vcore * C`
`Tp = (t-T) / C`
`Vmin = F * c`
`Vmin < Vcore`

stage 1 baseline equation
`Hashrate = frequency * C`

stage 2 min vcore
if min vcore not reached freq will be rejected or very low 
`Vcore > Vmin`
`let Vmin = frequency  * c`
`let e = 1e-7`
`Pv = Vcore-Vmin/(abs(Vcore-Vmin)+e)`
`penatly = 1-e when Vcore>Vmin`
`penatly = e-1 when Vmin>Vcore`

stage 3 temperatture
`t = Vcore * C`
`Pt = (t0-t) / C`



simplifying
`Hashrate = Pt * Pv * frequency * C`
`Hashrate = (t0- Vcore * C0) / C1 * (Vcore-(frequency  * C3)/(abs(Vcore-(frequency  * C3))+e)) * frequency * C2`

`Hashrate = (t0- Vcore * C0) / C1 * (Vcore-(frequency  * C3)/(abs(Vcore-(frequency  * C3))+e)) * frequency * C2`

`Hashrate = (t0- Vcore * C0) / C1 * (Vcore-(frequency  * C3)/(abs(Vcore-(frequency  * C3))+e)) * frequency * C2`
a* vocore * b*vcore-freq/abs(vcore-freq) * freq  = a* vcore * b*sign(vcore-freq) * freq


C2 is known when you hve smallcore count of the asic
t0 is not known 

t0 is found by inceasing Vcore until hashrate drop
C1 is also known when finding the hashrate drop as you will have examples  
C0 is also known when finding the hashrate drop as you will have examples  


Now we have a method for parameterising the Theoretical Hashrate equation 


differential for z
z = f(x, y)
dz = fx(x, y) · dx + fy(x, y) · dy
f (x + ∆x, y + ∆y) = f (x, y) + ∆z



Z
Z = (k_0 - y k_1)/k_2×(y - x k_3)/(abs(y - x k_3) + 1×10^(-7)) x k_4

'dZ/dF'
(10000000 k_4 (k_0 - k_1 y) ((y - 2 k_3 x) abs(y - x k_3) + 10000000 (y - k_3 x)^3))/(k_2 abs(y - x k_3) (10000000 abs(y - x k_3) + 1)^2)

'dZ/dVcore'
-(10000000 k_4 x (k_0 (-10000000 abs(y - x k_3)^2 - abs(y - x k_3) + 10000000 k_3^2 x^2 - 20000000 k_3 x y + 10000000 y^2) + k_1 (y (2 (10000000 abs(y - x k_3)^2 + abs(y - x k_3) - 5000000 y^2) - 10000000 k_3^2 x^2) - k_3 x (10000000 abs(y - x k_3)^2 + abs(y - x k_3) - 20000000 y^2))))/(k_2 abs(y - x k_3) (10000000 abs(y - x k_3) + 1)^2)

with the 'd/dF' and 'd/dVcore' we can do fun things we unlock first order gradient methods

we need to quickly calculate the coefficents {k0,...,k4}
each should have a mean and std make bounds [mean-2std,mean+2sd]
update std ?


Hashrate also has [mean-2std,mean+2sd]
update std ?


inital points 
3 particles [gradient based,global_based,new_locations]
we can have x particles with which tend towards current global and use the gradient function
each step update to wards 

multiple targets 
no targets

how to stop

noisy function evaluations















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
