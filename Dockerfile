# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

#Set ENV Variables
ENV MINER_IP=169.254.1.1
ENV MINER_VOLTS=1150
ENV MINER_FREQ=500
ENV VOLT_INC=25
ENV FREQ_INC=25
ENV BENCH_TIME=6
ENV SAM_INT=3
ENV MAX_ASIC_TEMP=75
ENV MAX_VOLT=1300
ENV MAX_FREQ=650
ENV MAX_VR_TEMP=90


# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY bitaxe_hashrate_benchmark.py .

RUN mkdir data

# Set the entrypoint
ENTRYPOINT ["python", "bitaxe_hashrate_benchmark.py"]
