import matplotlib.pyplot as plt
import numpy as np
from model import CityModel, plot_state_at_time

def run_once():
    # Simulation parameters

    # Define scale: 1 grid unit = 10 meters
    width = 100  # x10 meters wide
    height = 100  # x10 meters high

    num_tourists = 1000
    guided_ratio = 0.5  
    time_steps = 840  # 8 AM - 10 PM

    # Fixed Values
    num_pois = 8
    guided_start_times = [120, 360]  # 10 AM, 2 PM 
    self_guided_start_window = (0, 720)  # Self-guided tourists start between 8 AM and 8 PM
    self_guided_start_interval = 10  # New tourists start every 10-30 min within the window
    guided_wait_time = (10, 15)  
    self_guided_wait_time = (5, 10)  

    # Group Sizes (Now Configurable)
    guided_group_size = 20
    self_guided_group_size = 2

    np.random.seed(42)
    itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(num_pois)]
            
    # Run Simulation
    model = CityModel(width, height, num_tourists, guided_ratio, time_steps, guided_start_times, self_guided_start_window, self_guided_start_interval, guided_wait_time, self_guided_wait_time, guided_group_size, self_guided_group_size, itinerary)

    time_steps_2 = int(time_steps/2)
    
    for i in range(time_steps_2):
        model.step()        
        
    plot_state_at_time(model)
    
    for i in range(time_steps_2):
        model.step()

    # Collect Data
    data = model.datacollector.get_model_vars_dataframe()

    # Compute the mean congestion values
    avg_total_congestion = data["Total Congestion"].mean()
    avg_guided_congestion = data["Guided Congestion"].mean()
    avg_self_guided_congestion = data["Self-Guided Congestion"].mean()

    print(f"Average Total Congestion: {avg_total_congestion:.2f}")
    print(f"Average Guided Congestion: {avg_guided_congestion:.2f}")
    print(f"Average Self-Guided Congestion: {avg_self_guided_congestion:.2f}")   
        
    # Plots
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data["Guided Congestion"], c="orange", label="Guided Tourists")
    plt.plot(data["Self-Guided Congestion"], c="blue", label="Self-Guided Tourists")
    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Tourists at POIs")
    plt.title("Guided vs Self-Guided Congestion Over the Day")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(data["Guided Completion Rate"], c="orange", label="Guided Tourists")
    plt.plot(data["Self-Guided Completion Rate"], c="blue", label="Self-Guided Tourists")
    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Completion Rate (%)")
    plt.title("Guided vs Self-Guided Tour Completion Rate")
    plt.legend()

    plt.tight_layout()
    fig.savefig('output/eval.png', dpi=fig.dpi)
    plt.show()
    
    
def run_simulation(
    num_tourists,
    guided_group_size,
    self_guided_group_size,
    guided_wait_time,
    self_guided_wait_time):
    """Runs the simulation with given parameters and returns average congestion values."""
    
    width = 100  # x10 meters wide
    height = 100  # x10 meters high
    num_pois = 8
    time_steps = 840  # 8 AM - 10 PM
    
    np.random.seed(42)
    itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(num_pois)]
    
    model = CityModel(
        width=width, height=height, num_tourists=num_tourists, guided_ratio=0.5,
        total_time_steps=time_steps, guided_start_times=[120, 360],
        self_guided_start_window=(0, 720), self_guided_start_interval=10,
        guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
        guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
        itinerary=itinerary
    )

    for _ in range(time_steps):  # Simulate a full day
        model.step()

    # Extract congestion data
    data = model.datacollector.get_model_vars_dataframe()
    avg_total_congestion = data["Total Congestion"].mean()
    avg_guided_congestion = data["Guided Congestion"].mean()
    avg_self_guided_congestion = data["Self-Guided Congestion"].mean()

    return avg_total_congestion, avg_guided_congestion, avg_self_guided_congestion
    
    
def run_multi_eval(mode):
    # Parameter ranges for experiments
    group_sizes = list(range(5, 51, 5))  # Test guided group sizes from 5 to 50
    num_tourists_range = list(range(200, 2001, 200))
    num_tourists = 1000  # Fix number of tourists

    # Store results
    total_congestion_results = []
    guided_congestion_results = []
    self_guided_congestion_results = []
    label_ext = ""
    
    if mode == 1:
        for num_tourists in num_tourists_range:
            print("running num tourists: " + str(num_tourists))
            avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, 25, 2, (10, 15), (5, 10))  # Keep self-guided groups small
            total_congestion_results.append(avg_total)
            guided_congestion_results.append(avg_guided)
            self_guided_congestion_results.append(avg_self_guided)
        plot_x = num_tourists_range
        plot_x_label = "Number of Tourists"
        label_ext = "num_tourists"
    
    elif mode == 2:
        for size in group_sizes:
            print("running size: " + str(size))
            avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, size, size, (10, 15), (5, 10))  # Keep self-guided groups small
            total_congestion_results.append(avg_total)
            guided_congestion_results.append(avg_guided)
            self_guided_congestion_results.append(avg_self_guided)
        plot_x = group_sizes
        plot_x_label = "Group Size"
        label_ext = "group_size"
    
    elif mode == 3:
        waiting_times = list(range(5, 51, 5))
        for wtime in waiting_times:
            print("running wtime: " + str(wtime))
            avg_total, avg_guided, avg_self_guided = run_simulation(num_tourists, 25, 2, (wtime, wtime+1), (wtime, wtime+1))  # Keep self-guided groups small
            total_congestion_results.append(avg_total)
            guided_congestion_results.append(avg_guided)
            self_guided_congestion_results.append(avg_self_guided)
        plot_x = waiting_times        
        plot_x_label = "Waiting time at POIs"
        label_ext = "waiting_time"

    # Plot results
    fig = plt.figure(figsize=(10, 5))
    plt.plot(plot_x, guided_congestion_results, marker='o', c="orange", label="Guided Tour Congestion")
    plt.plot(plot_x, self_guided_congestion_results, marker='s', c="blue", label="Self-Guided Tour Congestion")
    # plt.plot(group_sizes, total_congestion_results, marker='x', label="Overall Tour Congestion")
    plt.xlabel(plot_x_label)
    plt.ylabel("Average Congestion")
    plt.title("Effect of " + plot_x_label + " on Congestion")
    plt.legend()
    plt.grid(True)
    fig.savefig('output/multi_eval_' + label_ext + '.png', dpi=fig.dpi)
    plt.show()


# run_once()
run_multi_eval(1)
# run_multi_eval(2)
# run_multi_eval(3)