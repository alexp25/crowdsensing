from model import BeliefModel
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def run_once():
    model = BeliefModel(
        width=10,
        height=10,
        dk_ratio=0.5,
        malicious_ratio=0.2,
        influencer_ratio=0.1,
        influence_rate=0.1,
        learning_rate=0.05
    )

    for _ in range(50):
        model.step()

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(model.data_belief, label="Average Belief")
    plt.axhline(y=0.75, color='red', linestyle='--', label="Tipping Threshold")
    plt.title("Belief Evolution (Mesa 3.1.4 Compatible)")
    plt.xlabel("Step")
    plt.ylabel("Average Belief")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def run_multi():
    # Simulation and plotting
    dk_values = [0.1, 0.3, 0.5]
    malicious_values = [0.1, 0.3, 0.5]
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan', 'magenta', 'black']

    plt.figure(figsize=(12, 6))

    for i, dk in enumerate(dk_values):
        for j, mal in enumerate(malicious_values):
            model = BeliefModel(
                width=10,
                height=10,
                dk_ratio=dk,
                malicious_ratio=mal,
                influencer_ratio=0.1,
                influence_rate=0.1,
                learning_rate=0.05
            )

            for _ in range(50):
                model.step()

            label = f"DK: {dk:.1f}, Mal: {mal:.1f}"
            color = colors[i * len(malicious_values) + j]
            plt.plot(model.data_belief, label=label, color=color)

    plt.axhline(y=0.75, color='red', linestyle='--', label="Tipping Threshold")
    plt.title("Belief Evolution for Multiple DK and Malicious Ratios")
    plt.xlabel("Step")
    plt.ylabel("Average Belief")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def run_heatmap():
    # Generate heatmap for DK ratio vs Malicious ratio with 10% influencers
    dk_values = np.linspace(0.1, 0.5, 9)
    mal_values = np.linspace(0.1, 0.5, 9)
    heatmap = np.zeros((len(dk_values), len(mal_values)))

    for i, dk in enumerate(dk_values):
        for j, mal in enumerate(mal_values):
            model = BeliefModel(dk_ratio=dk, malicious_ratio=mal, influencer_ratio=0.1)
            for _ in range(50):
                model.step()
            result = np.mean(model.data_belief)
            heatmap[i, j] = result

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin="lower", cmap="coolwarm", extent=[0.1, 0.5, 0.1, 0.5], aspect='auto')
    plt.colorbar(label="Final Average Belief")
    plt.xlabel("Malicious Agent Ratio")
    plt.ylabel("D-K Agent Ratio")
    plt.title("Final Average Belief vs DK Ratio & Malicious Ratio (with 10% Influencers)")
    plt.grid(False)
    plt.show()
    
    
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
            for _ in range(50):
                model.step()
            result = np.mean(model.data_belief)
            Z[i, j] = result

    # Plot 3D surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(DK, MAL, Z, cmap='viridis', edgecolor='k', linewidth=0.5)
    ax.set_title("Final Average Belief vs DK and Malicious Ratios (10% Influencers)")
    ax.set_xlabel("D-K Agent Ratio")
    ax.set_ylabel("Malicious Agent Ratio")
    ax.set_zlabel("Final Average Belief")
    plt.tight_layout()
    plt.savefig("belief_surface.png", dpi=300)
    print("Plot saved to belief_surface.png")
    
# run_once()
run_multi()
# run_heatmap()
# run_surface()