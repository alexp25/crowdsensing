import matplotlib.pyplot as plt
import numpy as np
from model import CityModel, plot_state_at_time
from model_ext import CityModelExt
from router import Router
from router import get_router
    
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

N_STEP = 50
belief_cmap = LinearSegmentedColormap.from_list(
    "belief_map", [(0, 0.5, 1.0), (1.0, 0.5, 0.2)], N=256
)
belief_norm = Normalize(vmin=0.5, vmax=0.9)


def get_heatmap_style():
    # Define custom colormap
    colors = [
        (0.5, 0.5, 1),  # Blue
        (0.5, 0.5, 1),  # Still blue at 0.5
        (1, 0.5, 0.2)   # Red at 1.0
    ]
    positions = [0.0, 0.5, 1.0]  # Map full range from 0 to 1
    cmap = LinearSegmentedColormap.from_list("sharp_halfblue_to_red", list(zip(positions, colors)))

    # Normalize full range
    norm = Normalize(vmin=0.5, vmax=0.9)
    return cmap, norm


def run_once():
    # Simulation parameters

    # Define scale: 1 grid unit = 10 meters
    width = 100  # x10 meters wide
    height = 100  # x10 meters high

    num_tourists = 1000
    guided_ratio = 0.5  
    time_steps = 1.0 * 840  # 8 AM - 10 PM

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
    # model = CityModel(width, height, num_tourists, guided_ratio, time_steps, guided_start_times, self_guided_start_window, self_guided_start_interval, guided_wait_time, self_guided_wait_time, guided_group_size, self_guided_group_size, itinerary)

    if not USE_MODEL_EXT: 
        model = CityModel(
            width=width, height=height, num_tourists=num_tourists, guided_ratio=guided_ratio,
            total_time_steps=time_steps, guided_start_times=guided_start_times,
            self_guided_start_window=self_guided_start_window, self_guided_start_interval=self_guided_start_interval,
            guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
            guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
            itinerary=itinerary
        )
    else:        
        # Build the router once
        router = get_router("data/walk_oldtown.graphml", center_latlon=(41.3826, 2.1769), dist_m=1000)
   
        model = CityModelExt(
            width=width, height=height, num_tourists=num_tourists, guided_ratio=0.5,
            total_time_steps=time_steps, guided_start_times=[120, 360],
            self_guided_start_window=(0, 720), self_guided_start_interval=10,
            guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
            guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
            itinerary=[],               # not used for routing anymore
            router = router,
            poi_capacity_range=(25, 60),
            clustered=False, num_clusters=2, cluster_radius=8
        )
        
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
    avg_total_queued = data["Total Queued"].mean()

    print(f"Average Total Congestion: {avg_total_congestion:.2f}")
    print(f"Average Guided Congestion: {avg_guided_congestion:.2f}")
    print(f"Average Self-Guided Congestion: {avg_self_guided_congestion:.2f}")   
    print(f"Average Queued: {avg_total_queued:.2f}")   
        
    # Plots
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data["Guided Congestion"], c="orange", label="Guided Tourists")
    plt.plot(data["Self-Guided Congestion"], c="blue", label="Self-Guided Tourists")
    plt.plot(data["Total Queued"], c="green", label="Tourists in Queue")
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
    
    global USE_MODEL_EXT
    
    width = 100  # x10 meters wide
    height = 100  # x10 meters high
    num_pois = 8
    time_steps = 840  # 8 AM - 10 PM
    
    np.random.seed(42)
    itinerary = [(np.random.randint(1, width - 1), np.random.randint(1, height - 1)) for _ in range(num_pois)]
   
    if not USE_MODEL_EXT: 
        model = CityModel(
            width=width, height=height, num_tourists=num_tourists, guided_ratio=0.5,
            total_time_steps=time_steps, guided_start_times=[120, 360],
            self_guided_start_window=(0, 720), self_guided_start_interval=10,
            guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
            guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
            itinerary=itinerary
        )
    else:       
        router = get_router("data/walk_oldtown.graphml", center_latlon=(41.3826, 2.1769), dist_m=1000)
   
        model = CityModelExt(
            width=width, height=height, num_tourists=num_tourists, guided_ratio=0.5,
            total_time_steps=time_steps, guided_start_times=[120, 360],
            self_guided_start_window=(0, 720), self_guided_start_interval=10,
            guided_wait_time=guided_wait_time, self_guided_wait_time=self_guided_wait_time,
            guided_group_size=guided_group_size, self_guided_group_size=self_guided_group_size,
            itinerary=[],               # not used for routing anymore
            router = router,
            poi_capacity_range=(2500, 6000)
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



def run_heatmap_congestion_vs_load_and_distribution(
    num_tourists_values = list(range(200, 2001, 200)),  # Y-axis
    guided_ratios = np.linspace(0.0, 1.0, 11),          # X-axis (0%..100% guided)
    guided_group_size: int = 25,
    self_guided_group_size: int = 2,
    guided_wait_time=(10, 15),
    self_guided_wait_time=(5, 10),
    width: int = 100,
    height: int = 100,
    num_pois: int = 8,
    time_steps: int = 840,                  # full day, 1-minute steps
    guided_start_times=[120, 360],          # 10:00, 14:00
    self_guided_start_window=(0, 720),      # 08:00–20:00
    self_guided_start_interval=10,
    seed: int = 42,
    out_prefix: str = "sa_congestion_load_vs_distribution"
):
    """
    Sensitivity analysis heatmaps:
      X: guided_ratio (0..1)
      Y: total number of tourists
    Cells: average congestion metric over the day.

    Saves three figures:
      1) total congestion
      2) guided congestion
      3) self-guided congestion

    Returns dict of matrices for further analysis.
    """
    X = np.array(guided_ratios)                 # columns
    Y = np.array(num_tourists_values)           # rows
    nx, ny = len(X), len(Y)

    total_cong = np.zeros((ny, nx))
    guided_cong = np.zeros((ny, nx))
    self_cong   = np.zeros((ny, nx))

    # Fixed itinerary across the entire grid for fairness
    rng = np.random.RandomState(seed)
    itinerary = [(rng.randint(1, width - 1), rng.randint(1, height - 1)) for _ in range(num_pois)]

    for iy, n_tour in enumerate(Y):
        for ix, gratio in enumerate(X):
            model = CityModel(
                width=width, height=height,
                num_tourists=int(n_tour),
                guided_ratio=float(gratio),
                total_time_steps=time_steps,
                guided_start_times=guided_start_times,
                self_guided_start_window=self_guided_start_window,
                self_guided_start_interval=self_guided_start_interval,
                guided_wait_time=guided_wait_time,
                self_guided_wait_time=self_guided_wait_time,
                guided_group_size=int(guided_group_size),
                self_guided_group_size=int(self_guided_group_size),
                itinerary=itinerary
            )
            for _ in range(time_steps):
                model.step()

            df = model.datacollector.get_model_vars_dataframe()
            total_cong[iy, ix]  = df["Total Congestion"].mean()
            guided_cong[iy, ix] = df["Guided Congestion"].mean()
            self_cong[iy, ix]   = df["Self-Guided Congestion"].mean()

    # Plot helper
    def _plot(matrix, title, cbar_label, fname):
        cmap, _ = get_heatmap_style()
        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
        norm = Normalize(vmin=vmin, vmax=vmax)

        plt.figure(figsize=(10, 7))
        plt.imshow(
            matrix,
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=[X.min(), X.max(), Y.min(), Y.max()],
            aspect="auto"
        )
        cbar = plt.colorbar()
        cbar.set_label(cbar_label, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        # cbar.ax.set_title(f"{title}", fontsize=14)    

        plt.tick_params(labelsize=14)
        # Axis labels & ticks
        plt.xlabel("Guided Ratio (% of tourists)", fontsize=14)
        xticks = np.linspace(X.min(), X.max(), 6)
        plt.xticks(xticks, [f"{int(x*100)}%" for x in xticks])

        plt.ylabel("Number of Tourists", fontsize=14)
        # choose nice ticks for Y (multiples of 200 by default)
        ystep = max(1, (Y.max() - Y.min()) // 8 // 100) * 100
        yvals = np.arange(Y.min(), Y.max() + ystep, ystep)
        plt.yticks(yvals)

        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig("output/" + fname, dpi=300)

        print(f"Saved: {fname}")
        plt.close()

    _plot(
        total_cong,
        "Average Congestion Sensitivity Analysis",
        "Average Total Congestion (tourists at POIs)",
        f"{out_prefix}_total.png"
    )

    return {
        "X_guided_ratios": X,
        "Y_num_tourists": Y,
        "total_congestion": total_cong,
        "guided_congestion": guided_cong,
        "self_congestion": self_cong
    }


USE_MODEL_EXT = True

run_once()
# run_multi_eval(1)
# run_multi_eval(2)
# run_multi_eval(3)
# run_heatmap()
# run_heatmap_congestion_vs_load_and_distribution()