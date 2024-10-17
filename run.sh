#!/bin/bash
export ROOT_PATH='Revolve'
export AIRSIM_PATH='AirSim'
export AIRSIMNH_PATH='AirSimNH/AirSimNH/LinuxNoEditor/AirSimNH.sh'
export OPENAI_API_KEY=''

# Start the simulations in the background
python3 start_sims.py &

# Store the PID of the background process
SIM_PID=$!

echo "Simulations are starting in the background. PID: $SIM_PID"

# Wait for a bit to allow simulations to initialize
echo "Waiting for simulations to initialize..."
sleep 90  # Adjust this sleep time as necessary

# Now, run the check script
echo "Running check script..."
#python3 check_sims.py
python3 revolve_auto.py

echo "Terminating simulations..."
killall -u <user_name>

