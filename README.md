Smart Traffic Light - SUMO Simulation

This repository contains the code and configuration for a Smart Traffic Light system, developed using the SUMO (Simulation of Urban MObility) simulator. The project focuses on the extraction, processing, and analysis of vehicular flow data to model strategies that optimize waiting times and reduce congestion in urban intersections.

Project Objectives

- Realistic Traffic Simulation: Model complex intersections and vehicular flows using micro-simulation tools.
- Time Optimization: Evaluate traffic light control algorithms to minimize vehicle idle time.
- Data Extraction and Analysis: Collect performance metrics (waiting times, queue lengths, emissions) for subsequent structured analysis.

Technologies and Tools

- SUMO (Simulation of Urban MObility): Main engine for traffic micro-simulation.
- TraCI (Traffic Control Interface): Python interface to interact with the SUMO simulation in real time.
- Python: Main language for control logic, simulation automation, and data collection.
- Pandas & Matplotlib/Seaborn: Tools used for processing, cleaning, and visualizing the data extracted after the simulations.

Project Structure

semaforo-sumo/
- data/ : Output data and simulation results (.csv, .xml)
- networks/ : SUMO network files (.net.xml)
- routes/ : Vehicle demand and route files (.rou.xml)
- scripts/ : Python scripts for analysis and charts
- src/ : Main source code (TraCI control logic)
  - main_controller.py : Main execution script
- sumo_config.sumocfg : Main SUMO configuration file
- README.txt

Installation and Prerequisites

To run this simulation, you need to have SUMO and Python installed on your system (compatible with Linux Debian/Ubuntu and Windows environments).

1. Install SUMO:
   - On Debian/Ubuntu: sudo apt-get install sumo sumo-tools sumo-doc
   - Make sure to configure the SUMO_HOME environment variable.
2. Clone the repository:
   git clone https://github.com/YourUsername/semaforo-sumo.git
   cd semaforo-sumo
3. Install Python dependencies:
   pip install traci pandas matplotlib

Usage

To start the simulation with the SUMO graphical interface (sumo-gui) and the intelligent control logic, run:

python src/main_controller.py

If you want to run the simulation in the background without a graphical interface (for massive data collection), change the sumo-gui parameter to sumo inside the main script.

Results Analysis

At the end of each run, the system generates log files and a structured dataset (CSV) in the data/ folder. The scripts in the scripts/ folder allow you to generate visualizations comparing:
- The performance of the traditional static traffic light vs. the adaptive smart traffic light.
- The percentage reduction in waiting times per lane.
