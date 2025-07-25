from model import BeliefModel
import numpy as np
import networkx as nx
import matplotlib
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


def grid_to_circle_positions(width, height):
    """
    Map (x, y) grid coordinates to a circular surface using polar normalization.
    """
    cx, cy = width / 2, height / 2  # center
    radius = min(width, height) / 2
    positions = {}

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            dx = x - cx
            dy = y - cy
            norm_r = np.sqrt(dx**2 + dy**2) / radius
            if norm_r > 1:
                norm_r = 1  # clip to edge of circle
            angle = np.arctan2(dy, dx)
            r = norm_r
            px = r * np.cos(angle)
            py = r * np.sin(angle)
            positions[idx] = (px, py)

    return positions


def plot_belief_state_from_model(model, step=None):
    """
    Plots agent beliefs in a circular layout with color-coded nodes and type labels.
    Malicious = Red, Influencer = Blue, Normal = Belief-shaded
    """  

    title="Belief Network"
    G = nx.Graph()
    pos = {}
    node_colors = []
    node_sizes = []
    agent_positions = {}
    width = model.width
    height = model.height

    for idx, agent in enumerate(model.agents):
        x = idx % model.width
        y = idx // model.height
        pos[idx] = (x, y)
        agent_positions[(x, y)] = idx

        G.add_node(idx)

        node_sizes.append(200)

        if agent.type == "malicious":
            node_colors.append("red")
        elif agent.type == "influencer":
            node_colors.append("blue")
        else:
            node_colors.append(belief_cmap(belief_norm(agent.belief)))

    fig, ax = plt.subplots(figsize=(8, 8))
    # plt.subplots_adjust(right=0.4, top=0.85, bottom=0.4)    
    
    # Add edges for grid neighbors (4-neighborhood)
    for (x, y), idx in agent_positions.items():
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < width and 0 <= ny_ < height:
                neighbor_idx = agent_positions.get((nx_, ny_))
                if neighbor_idx is not None:
                    G.add_edge(idx, neighbor_idx)

    # Draw network and grid edges
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, edge_color="gray", linewidths=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='k', label='Malicious'),
        Patch(facecolor='blue', edgecolor='k', label='Corrective'),
        # Patch(facecolor='lightgray', edgecolor='k', label='Normal (belief color)')
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=[1.02, 0.98],
        fontsize=11,
        title='Agent Types',
        title_fontsize=12,
        frameon = True
    )

    # Colorbar for belief values
    sm = ScalarMappable(cmap=belief_cmap, norm=belief_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, shrink=0.9, aspect=35)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Normal / DK Agents (Belief)", fontsize=12)

    # Layout and title
    # ax.set_title(f"{title} (Step {step})" if step is not None else title, fontsize=14)
    ax.set_title(f"{title}", fontsize=14)    
    plt.tight_layout()
    
    filename = "belief_graph_fixed_step_" + str(step) + ".png"
    plt.savefig(filename, dpi=300)
    print("Plot saved to " + filename)
    # plt.show()


def run_once():
    model = BeliefModel(
        width=10,
        height=10,
        dk_ratio=0.2,
        malicious_ratio=0.2,
        influencer_ratio=0.1,
        # influence_rate=0.1,
        # learning_rate=0.05
    )
    
    plot_belief_state_from_model(model, step=0)

    for _ in range(500):
        model.step()    

    plot_belief_state_from_model(model, step=500)
     
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(model.data_belief, label="Average Belief")
    plt.axhline(y=0.75, color='red', linestyle='--', label="Tipping Threshold")
    plt.title("Belief Evolution (Mesa 3.1.4 Compatible)")
    plt.xlabel("Step")
    plt.ylabel("Average Belief")
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    filename = "belief_plot_fixed.png"
    plt.savefig(filename, dpi=300)
    print("Plot saved to " + filename)
    
    
def run_multi():
    # Simulation and plotting
    dk_values = [0.1, 0.3, 0.5]
    # dk_values = [0.3]
    malicious_values = [0.1, 0.3, 0.5]
    # malicious_values = [0.3]
    infl_value = 0.1
    
    # colors = ['blue', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan', 'magenta', 'black']
    
    # colors = ['blue', 'blue', 'blue', 'orange', 'orange', 'orange', 'red', 'red', 'red']
    # colors = ['blue', 'orange', 'red', 'blue', 'orange', 'red', 'blue', 'orange', 'red']
    
    colors = [
        '#aec7e8', '#ffbb78', '#ff9999',  # Light blue/orange/red for DK 0.1
        '#1f77b4', '#ff7f0e', '#d62728',  # Medium (default) shades for DK 0.3
        '#084594', '#e6550d', '#99000d'   # Dark blue/orange/red for DK 0.5
    ]

    plt.figure(figsize=(12, 6))

    for i, dk in enumerate(dk_values):
        for j, mal in enumerate(malicious_values):
            model = BeliefModel(
                dk_ratio=dk,
                malicious_ratio=mal,
                influencer_ratio=infl_value
            )

            for _ in range(N_STEP*4):
                model.step()

            label = f"DK: {dk:.1f}, Mal: {mal:.1f}"
            color = colors[i * len(malicious_values) + j]
            plt.plot(model.data_belief, label=label, color=color)

    # plt.axhline(y=0.75, color='red', linestyle='--', label="Tipping Threshold")
    plt.axhline(y=0.75, color='red', linestyle='--')
    plt.title("Belief Evolution for Multiple DK and Malicious Ratios")
    plt.xlabel("Step")
    plt.ylabel("Average Belief")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    filename = "belief_plot_fixed_influencers_" + str(int(infl_value * 100)) + ".png"
    plt.savefig(filename, dpi=300)
    print("Plot saved to " + filename)
    
    
def run_heatmap_fixed_infl():
    # Generate heatmap for DK ratio vs Malicious ratio with 10% influencers
    dk_values = np.linspace(0.1, 0.5, 9)
    mal_values = np.linspace(0.1, 0.5, 9)
    heatmap = np.zeros((len(dk_values), len(mal_values)))
    infl_value = 0.1

    for i, dk in enumerate(dk_values):
        for j, mal in enumerate(mal_values):
            model = BeliefModel(dk_ratio=dk, malicious_ratio=mal, influencer_ratio=infl_value)
            for _ in range(N_STEP):
                model.step()
            result = np.mean(model.data_belief)
            heatmap[i, j] = result

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    # plt.imshow(heatmap, origin="lower", cmap="coolwarm", extent=[0.1, 0.5, 0.1, 0.5], aspect='auto')
    
    cmap, norm = get_heatmap_style()
    plt.imshow(heatmap, origin="lower", cmap=cmap, norm=norm, extent=[0.1, 0.5, 0.1, 0.5], aspect='auto')

    plt.colorbar(label="Final Average Belief")
    plt.xlabel("Malicious Agent Ratio")
    plt.ylabel("D-K Agent Ratio")
    plt.title("Final Average Belief with " + str(int(infl_value * 100)) + "% corrective agents")
    plt.grid(False)
    # plt.show()
    
    filename = "belief_heatmap_fixed_influencers_" + str(int(infl_value * 100)) + ".png"
    plt.savefig(filename, dpi=300)
    print("Plot saved to " + filename)
    
  
def run_heatmap_fixed_mal():
    # Generate heatmap for DK ratio vs Malicious ratio with 10% influencers
    dk_values = np.linspace(0.1, 0.5, 9)
    infl_values = np.linspace(0.1, 0.5, 9)
    heatmap = np.zeros((len(dk_values), len(infl_values)))
    mal_value = 0.3

    for i, dk in enumerate(dk_values):
        for j, infl in enumerate(infl_values):
            model = BeliefModel(dk_ratio=dk, malicious_ratio=mal_value, influencer_ratio=infl)
            for _ in range(N_STEP):
                model.step()
            result = np.mean(model.data_belief)
            heatmap[i, j] = result

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    # plt.imshow(heatmap, origin="lower", cmap="coolwarm", extent=[0.1, 0.5, 0.1, 0.5], aspect='auto')
    
    cmap, norm = get_heatmap_style()
    plt.imshow(heatmap, origin="lower", cmap=cmap, norm=norm, extent=[0.1, 0.5, 0.1, 0.5], aspect='auto')
    
    plt.colorbar(label="Final Average Belief")
    plt.xlabel("Corrective Agent Ratio")
    plt.ylabel("D-K Agent Ratio")
    plt.title("Final Average Belief with " + str(int(mal_value * 100)) + "% Malicious")
    plt.grid(False)
    # plt.show()
    
    filename = "belief_heatmap_fixed_mal_" + str(int(mal_value * 100)) + ".png"   
    plt.savefig(filename, dpi=300)
    print("Plot saved to " + filename)
   
    
def run_surface():
    matplotlib.use('Agg')  # For headless (no window) mode
    # or try this instead for interactive:
    # matplotlib.use('QtAgg')  # if you have PyQt5 or PySide2 installed
    # Prepare data for 3D plot
    dk_values = np.linspace(0.1, 0.5, 9)
    mal_values = np.linspace(0.1, 0.5, 9)
    DK, MAL = np.meshgrid(dk_values, mal_values)
    Z = np.zeros_like(DK)

    # Run simulations
    for i in range(DK.shape[0]):
        for j in range(DK.shape[1]):
            model = BeliefModel(dk_ratio=DK[i, j], malicious_ratio=MAL[i, j], influencer_ratio=0.1)
            for _ in range(N_STEP):
                model.step()
            result = np.mean(model.data_belief)
            Z[i, j] = result

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(DK, MAL, Z, cmap='viridis', edgecolor='k', linewidth=0.5)
    ax.set_title("Final Average Belief with 10% corrective agents")
    ax.set_xlabel("D-K Agent Ratio")
    ax.set_ylabel("Malicious Agent Ratio")
    ax.set_zlabel("Final Average Belief")
    plt.tight_layout()
    plt.savefig("belief_surface.png", dpi=300)
    print("Plot saved to belief_surface.png")
    
# run_once()
run_multi()
# run_heatmap_fixed_infl()
# run_heatmap_fixed_mal()
# run_surface()