import requests
import time
import json
import signal
import sys
import os

# ANSI Color Codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# Replace the configuration section
miner_ip = os.environ['MINER_IP']
bitaxe_ip = f"http://{miner_ip}"
initial_voltage = int(os.environ['MINER_VOLTS'])
initial_frequency = int(os.environ['MINER_FREQ'])

if miner_ip == "169.254.1.1":
        print(f"IP address of the Bitaxe  {miner_ip}")
        print(f"Voltage {initial_voltage}")
        print(f"Frequency {initial_frequency}")
        print(f"  MINER_IP is still default, please change")
        sys.exit(1)


print(f"IP address of the Bitaxe  {miner_ip}")
print(f"Voltage {initial_voltage}")
print(f"Frequency {initial_frequency}")

# Configuration
#voltage_increment =25
#frequency_increment = 25
#benchmark_time = 60   # 20 minutes benchmark time
#sample_interval = 3   # 30 seconds sample interval
#max_temp = 75         # Will stop if temperature reaches or exceeds this value
#max_allowed_voltage = 1300
#max_allowed_frequency = 650
#max_vr_temp = 90  # Maximum allowed voltage regulator temperature
voltage_increment = int(os.environ['VOLT_INC'])
frequency_increment = int(os.environ['FREQ_INC'])
benchmark_time = int(os.environ['BENCH_TIME'])   # 20 minutes benchmark time
sample_interval = int(os.environ['SAM_INT'])   # 3 = 30 seconds sample interval
max_temp = int(os.environ['MAX_ASIC_TEMP'])      # Will stop if temperature reaches or exceeds this value in C
max_allowed_voltage = int(os.environ['MAX_VOLT'])
max_allowed_frequency = int(os.environ['MAX_FREQ'])
max_vr_temp = int(os.environ['MAX_VR_TEMP'])  # Maximum allowed voltage regulator temperature in C

# Add these variables to the global configuration section
small_core_count = None
asic_count = None

# Validate core voltages
if initial_voltage > max_allowed_voltage:
    raise ValueError(RED + f"Error: Initial voltage exceeds the maximum allowed value of {max_allowed_voltage}mV. Please check the input and try again." + RESET)

# Validate frequency
if initial_frequency > max_allowed_frequency:
    raise ValueError(RED + f"Error: Initial frequency exceeds the maximum allowed value of {max_allowed_frequency}Mhz. Please check the input and try again." + RESET)

# Results storage
results = []

# Dynamically determined default settings
default_voltage = None
default_frequency = None

# Check if we're handling an interrupt (Ctrl+C)
handling_interrupt = False

def fetch_default_settings():
    global default_voltage, default_frequency, small_core_count, asic_count
    try:
        response = requests.get(f"{bitaxe_ip}/api/system/info", timeout=10)
        response.raise_for_status()
        system_info = response.json()
        default_voltage = system_info.get("coreVoltage", 1150)  # Fallback to 1150 if not found
        default_frequency = system_info.get("frequency", 500)  # Fallback to 500 if not found
        small_core_count = system_info.get("smallCoreCount", 0)
        asic_count = system_info.get("asicCount", 0)
        print(GREEN + f"Current settings determined:\n"
                      f"  Core Voltage: {default_voltage}mV\n"
                      f"  Frequency: {default_frequency}MHz\n"
                      f"  ASIC Configuration: {small_core_count * asic_count} total cores" + RESET)
    except requests.exceptions.RequestException as e:
        print(RED + f"Error fetching default system settings: {e}. Using fallback defaults." + RESET)
        default_voltage = 1150
        default_frequency = 500
        small_core_count = 0
        asic_count = 0
    return {'v':default_voltage,'f':default_frequency,'cores':small_core_count,'chips':asic_count}

# Add a global flag to track whether the system has already been reset
system_reset_done = False

def handle_sigint(signum, frame):
    global system_reset_done, handling_interrupt
    
    # If we're already handling an interrupt or have completed reset, ignore this signal
    if handling_interrupt or system_reset_done:
        return
        
    handling_interrupt = True
    print(RED + "Benchmarking interrupted by user." + RESET)
    
    try:
        if results:
            reset_to_best_setting()
            save_results()
            print(GREEN + "Bitaxe reset to best or default settings and results saved." + RESET)
        else:
            print(YELLOW + "No valid benchmarking results found. Applying predefined default settings." + RESET)
            set_system_settings(default_voltage, default_frequency)
    finally:
        system_reset_done = True
        handling_interrupt = False
        sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_sigint)

def get_system_info():
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(f"{bitaxe_ip}/api/system/info", timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.Timeout:
            print(YELLOW + f"Timeout while fetching system info. Attempt {attempt + 1} of {retries}." + RESET)
        except requests.exceptions.ConnectionError:
            print(RED + f"Connection error while fetching system info. Attempt {attempt + 1} of {retries}." + RESET)
        except requests.exceptions.RequestException as e:
            print(RED + f"Error fetching system info: {e}" + RESET)
            break
        time.sleep(5)  # Wait before retrying
    return None

def set_system_settings(core_voltage, frequency):
    settings = {
        "coreVoltage": core_voltage,
        "frequency": frequency
    }
    try:
        response = requests.patch(f"{bitaxe_ip}/api/system", json=settings, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(YELLOW + f"Applying settings: Voltage = {core_voltage}mV, Frequency = {frequency}MHz" + RESET)
        time.sleep(2)
        restart_system()
    except requests.exceptions.RequestException as e:
        print(RED + f"Error setting system settings: {e}" + RESET)

def restart_system():
    try:
        print(YELLOW + "Restarting Bitaxe system to apply new settings..." + RESET)
        response = requests.post(f"{bitaxe_ip}/api/system/restart", timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        time.sleep(15)  # Allow 60s time for the system to restart and start hashing
    except requests.exceptions.RequestException as e:
        print(RED + f"Error restarting the system: {e}" + RESET)

def benchmark_iteration(core_voltage, frequency):
    print(GREEN + f"Starting benchmark for Core Voltage: {core_voltage}mV, Frequency: {frequency}MHz" + RESET)
    hash_rates = []
    temperatures = []
    power_consumptions = []
    total_samples = benchmark_time // sample_interval
    expected_hashrate = frequency * ((small_core_count * asic_count) / 1000)  # Calculate expected hashrate based on frequency
    
    for sample in range(total_samples):
        info = get_system_info()
        if info is None:
            print(YELLOW + "Skipping this iteration due to failure in fetching system info." + RESET)
            return None, None, None, False
        
        temp = info.get("temp")
        vr_temp = info.get("vrTemp")  # Get VR temperature if available
        
        if temp is None:
            print(YELLOW + "Temperature data not available." + RESET)
            return None, None, None, False
        
        # Check both chip and VR temperatures
        if temp >= max_temp:
            print(RED + f"Chip temperature exceeded {max_temp}°C! Stopping current benchmark." + RESET)
            return None, None, None, False
            
        if vr_temp is not None and vr_temp >= max_vr_temp:
            print(RED + f"Voltage regulator temperature exceeded {max_vr_temp}°C! Stopping current benchmark." + RESET)
            return None, None, None, False
        
        hash_rate = info.get("hashRate")
        power_consumption = info.get("power")
        
        if hash_rate is None or power_consumption is None:
            print(YELLOW + "Hashrate or Watts data not available." + RESET)
            return None, None, None, False
        
        hash_rates.append(hash_rate)
        temperatures.append(temp)
        power_consumptions.append(power_consumption)
        
        # Calculate percentage progress
        percentage_progress = ((sample + 1) / total_samples) * 100
        status_line = (
            f"[{sample + 1:2d}/{total_samples:2d}] "
            f"{percentage_progress:5.1f}% | "
            f"V: {core_voltage:4d}mV | "
            f"F: {frequency:4d}MHz | "
            f"H: {int(hash_rate):4d} GH/s | "
            f"T: {int(temp):2d}°C"
        )
        if vr_temp is not None and vr_temp > 0:
            status_line += f" | VR: {int(vr_temp):2d}°C"
        print(YELLOW + status_line + RESET)
        
        # Only sleep if it's not the last iteration
        if sample < total_samples - 1:
            time.sleep(sample_interval)
    
    if hash_rates and temperatures and power_consumptions:
        # Remove 3 highest and 3 lowest hashrates in case of outliers
        sorted_hashrates = sorted(hash_rates)
        #trimmed_hashrates = sorted_hashrates[3:-3] 
        vals = 0
        ts = 0
        hrs = 0
        ps = 0
        for i,h in enumerate(hash_rates):
            if h > 0:
                ts+=temperatures[i]
                ps+=power_consumptions[i]
                hrs+=hash_rates[i]
                vals +=1

        if vals == 0: vals = 1
        average_hashrate = hrs / vals
        average_temperature = ts / vals
        average_power = ps / vals
        
        # Add protection against zero hashrate
        if average_hashrate > 0:
            efficiency_jth = average_power / (average_hashrate / 1_000)
        else:
            print(RED + "Warning: Zero hashrate detected, skipping efficiency calculation" + RESET)
            #we want this data
            return average_hashrate, average_temperature, 0, False
        
        # Calculate if hashrate is within 8% of expected
        hashrate_within_tolerance = (average_hashrate >= expected_hashrate * 0.92)
        
        print(GREEN + f"Average Hashrate: {average_hashrate:.2f} GH/s (Expected: {expected_hashrate:.2f} GH/s)" + RESET)
        print(GREEN + f"Average Temperature: {average_temperature:.2f}°C" + RESET)
        print(GREEN + f"Efficiency: {efficiency_jth:.2f} J/TH" + RESET)
        
        return average_hashrate, average_temperature, efficiency_jth, hashrate_within_tolerance
    else:
        print(YELLOW + "No Hashrate or Temperature or Watts data collected." + RESET)
        return None, None, None, False

def save_results():
    try:
        # Extract IP from bitaxe_ip global variable and remove 'http://'
        ip_address = bitaxe_ip.replace('http://', '')
        filename = f"./data/bitaxe_benchmark_results_{ip_address}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        print(GREEN + f"Results saved to {filename}" + RESET)
    except IOError as e:
        print(RED + f"Error saving results to file: {e}" + RESET)

def reset_to_best_setting():
    if not results:
        print(YELLOW + "No valid benchmarking results found. Applying predefined default settings." + RESET)
        set_system_settings(default_voltage, default_frequency)
    else:
        best_result = sorted(results, key=lambda x: x["averageHashRate"], reverse=True)[0]
        best_voltage = best_result["coreVoltage"]
        best_frequency = best_result["frequency"]

        print(GREEN + f"Applying the best settings from benchmarking:\n"
                      f"  Core Voltage: {best_voltage}mV\n"
                      f"  Frequency: {best_frequency}MHz" + RESET)
        set_system_settings(best_voltage, best_frequency)
    
    restart_system()

if __name__ == "__main__":
#   Main benchmarking process
    try:
        fetch_default_settings()
        
        # Add disclaimer
        print(RED + "\nDISCLAIMER:" + RESET)
        print("This tool will stress test your Bitaxe by running it at various voltages and frequencies.")
        print("While safeguards are in place, running hardware outside of standard parameters carries inherent risks.")
        print("Use this tool at your own risk. The author(s) are not responsible for any damage to your hardware.")
        print("\nNOTE: Ambient temperature significantly affects these results. The optimal settings found may not")
        print("work well if room temperature changes substantially. Re-run the benchmark if conditions change.\n")
        
        current_voltage = initial_voltage
        current_frequency = initial_frequency
        
        while current_voltage <= max_allowed_voltage and current_frequency <= max_allowed_frequency:
            set_system_settings(current_voltage, current_frequency)
            avg_hashrate, avg_temp, efficiency_jth, hashrate_ok = benchmark_iteration(current_voltage, current_frequency)
            
            if avg_hashrate is not None and avg_temp is not None and efficiency_jth is not None:
                results.append({
                    "coreVoltage": current_voltage,
                    "frequency": current_frequency,
                    "averageHashRate": avg_hashrate,
                    "averageTemperature": avg_temp,
                    "efficiencyJTH": efficiency_jth
                })
                
                if hashrate_ok:
                    # If hashrate is good, try increasing frequency
                    if current_frequency + frequency_increment <= max_allowed_frequency:
                        current_frequency += frequency_increment
                    else:
                        break  # We've reached max frequency with good results
                else:
                    # If hashrate is not good, go back one frequency step and increase voltage
                    if current_voltage + voltage_increment <= max_allowed_voltage:
                        current_voltage += voltage_increment
                        current_frequency -= frequency_increment  # Go back to one frequency step and retry
                        print(YELLOW + f"Hashrate to low compared to expected. Decreasing frequency to {current_frequency}MHz and increasing voltage to {current_voltage}mV" + RESET)
                    else:
                        break  # We've reached max voltage without good results
            else:
                # If we hit thermal limits or other issues, we've found the highest safe settings
                print(GREEN + "Reached thermal or stability limits. Stopping further testing." + RESET)
                break  # Stop testing higher values

            save_results()

    except Exception as e:
        print(RED + f"An unexpected error occurred: {e}" + RESET)
        if results:
            reset_to_best_setting()
            save_results()
        else:
            print(YELLOW + "No valid benchmarking results found. Applying predefined default settings." + RESET)
            set_system_settings(default_voltage, default_frequency)
            restart_system()
    finally:
        if not system_reset_done:
            if results:
                reset_to_best_setting()
                save_results()
                print(GREEN + "Bitaxe reset to best or default settings and results saved." + RESET)
            else:
                print(YELLOW + "No valid benchmarking results found. Applying predefined default settings." + RESET)
                set_system_settings(default_voltage, default_frequency)
                restart_system()
            system_reset_done = True

        # Print results summary only if we have results
        if results:
            # Sort results by averageHashRate in descending order and get the top 5
            top_5_results = sorted(results, key=lambda x: x["averageHashRate"], reverse=True)[:5]
            
            # Create a dictionary containing all results and top performers
            final_data = {
                "all_results": results,
                "top_performers": [
                    {
                        "rank": i,
                        "coreVoltage": result["coreVoltage"],
                        "frequency": result["frequency"],
                        "averageHashRate": result["averageHashRate"],
                        "averageTemperature": result["averageTemperature"],
                        "efficiencyJTH": result["efficiencyJTH"]
                    }
                    for i, result in enumerate(top_5_results, 1)
                ]
            }
            
            # Save the final data to JSON
            ip_address = bitaxe_ip.replace('http://', '')
            filename = f"bitaxe_benchmark_results_{ip_address}.json"
            with open(filename, "w") as f:
                json.dump(final_data, f, indent=4)
            
            print(GREEN + "Benchmarking completed." + RESET)
            if top_5_results:
                print(GREEN + "\nTop 5 Performing Settings:" + RESET)
                for i, result in enumerate(top_5_results, 1):
                    print(GREEN + f"\nRank {i}:" + RESET)
                    print(GREEN + f"  Core Voltage: {result['coreVoltage']}mV" + RESET)
                    print(GREEN + f"  Frequency: {result['frequency']}MHz" + RESET)
                    print(GREEN + f"  Average Hashrate: {result['averageHashRate']:.2f} GH/s" + RESET)
                    print(GREEN + f"  Average Temperature: {result['averageTemperature']:.2f}°C" + RESET)
                    print(GREEN + f"  Efficiency: {result['efficiencyJTH']:.2f} J/TH" + RESET)
            else:
                print(RED + "No valid results were found during benchmarking." + RESET)
