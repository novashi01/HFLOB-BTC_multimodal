import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Slider
import matplotlib

def read_data(file_path, scaling_factor=100000):
    """
    Read and process LOB (Limit Order Book) data from a CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing LOB data
    scaling_factor : int, optional
        Factor used to denormalize LOB data, defaults to 100,000

    Returns:
    --------
    tuple or None
        - lob_data : numpy.ndarray
            Denormalized LOB data
        - label_data : numpy.ndarray
            Label data for price movement prediction
        - relative_mid_price : numpy.ndarray
            Mid-price calculated from the order book
        - volatility : numpy.ndarray
            Market volatility indicators
    """
    # Read CSV file using Pandas
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return None

    # Extract LOB data column names
    lob_columns = [
        'ask10', 'ask_vol10', 'ask09', 'ask_vol09', 'ask08', 'ask_vol08',
        'ask07', 'ask_vol07', 'ask06', 'ask_vol06', 'ask05', 'ask_vol05',
        'ask04', 'ask_vol04', 'ask03', 'ask_vol03', 'ask02', 'ask_vol02',
        'ask01', 'ask_vol01', 'bid01', 'bid_vol01', 'bid02', 'bid_vol02',
        'bid03', 'bid_vol03', 'bid04', 'bid_vol04', 'bid05', 'bid_vol05',
        'bid06', 'bid_vol06', 'bid07', 'bid_vol07', 'bid08', 'bid_vol08',
        'bid09', 'bid_vol09', 'bid10', 'bid_vol10'
    ]

    # Check for missing columns
    missing_columns = set(lob_columns) - set(data.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return None

    # Extract and transpose LOB data to match original structure
    lob_data = data[lob_columns].values.T

    # Denormalize LOB data
    lob_data = lob_data * scaling_factor

    # Extract label data
    label_columns = ['label_1', 'label_2', 'label_3', 'label_5', 'label_10']
    missing_labels = set(label_columns) - set(data.columns)
    if missing_labels:
        print(f"Missing label columns: {missing_labels}")
        return None
    label_data = data[label_columns].values.T

    # Replace 0s with 2s in label data
    label_data[label_data == 0] = 2

    # Extract relative mid price
    if 'mid_price' not in data.columns:
        print("Missing 'mid_price' column")
        return None
    relative_mid_price = data['mid_price'].values
    volatility = data['market_phase'].values

    return lob_data, label_data, relative_mid_price, volatility

# Load data
file_path = "../data/trade_lob_multi6/processed_BTC_trade_01_19.csv"
result = read_data(file_path)
if result is None:
    exit(1)
lob_data, label_data, relative_mid_price, volatility = result

# Set up color mapping
colors = ['#0000FF', '#00FFFF', '#FFFFFF', '#FFFF00', '#FF0000']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create visualization layout
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
plt.subplots_adjust(bottom=0.2)
fig.patch.set_facecolor('#1C1C1C')

time_window = 300
total_frames = lob_data.shape[1]

# Create position slider
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, 'Position', 0, 100, valinit=0, valstep=1)

def update(val):
    """
    Update visualization based on slider position.
    
    Parameters:
    -----------
    val : float
        Slider position value between 0 and 100
    """
    i = int(val / 100 * total_frames)
    start_idx = max(0, i - time_window // 2)
    end_idx = min(total_frames, start_idx + time_window)

    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Calculate denormalized best bid/ask prices
    best_bid = lob_data[20, start_idx:end_idx]
    best_ask = lob_data[0, start_idx:end_idx]
    lob_mid_price = (best_bid + best_ask) / 2

    # Plot LOB-based mid price
    ax1.plot(range(start_idx, end_idx), lob_mid_price, color='white', linewidth=1, label='LOB Mid Price')

    # Plot trade-based mid price
    ax1.plot(range(start_idx, end_idx), relative_mid_price[start_idx:end_idx], color='yellow', linewidth=1,
             label='Trade-based Mid Price')

    # Plot bid data (indices 20-39, step 2)
    for j in range(20, 40, 2):
        price = lob_data[j, start_idx:end_idx]
        volume = lob_data[j + 1, start_idx:end_idx]
        ax1.scatter(range(start_idx, end_idx), price, c=volume, cmap=cmap, s=2, alpha=0.7)

    # Plot ask data (indices 0-19, step 2)
    for j in range(0, 20, 2):
        price = lob_data[j, start_idx:end_idx]
        volume = lob_data[j + 1, start_idx:end_idx]
        ax1.scatter(range(start_idx, end_idx), price, c=volume, cmap=cmap, s=2, alpha=0.7)

    # Configure main plot appearance
    ax1.set_facecolor('#1C1C1C')
    ax1.grid(True, color='#2C2C2C', linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', colors='white')
    ax1.set_title('LOB Visualization', color='white')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    ax1.legend(loc='upper right')

    # Plot label data
    labels = ['label_1', 'label_2', 'label_3', 'label_5', 'label_10']
    colors_list = {1: 'green', 2: 'white', 3: 'red'}  # Map label values to colors
    for idx, label in enumerate(labels):
        label_values = label_data[idx, start_idx:end_idx]
        colors_to_plot = [colors_list.get(int(val), 'black') for val in label_values]
        ax2.scatter(range(start_idx, end_idx), [idx] * (end_idx - start_idx),
                    c=colors_to_plot, s=2)

    # Configure label plot appearance
    ax2.set_facecolor('#1C1C1C')
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.tick_params(axis='both', colors='white')
    ax2.set_xlim(ax1.get_xlim())

    # Plot and configure volatility visualization
    ax3.plot(range(start_idx, end_idx), volatility[start_idx:end_idx], color='cyan', linewidth=1)
    ax3.set_facecolor('#1C1C1C')
    ax3.grid(True, color='#2C2C2C', linestyle='-', linewidth=0.5)
    ax3.tick_params(axis='both', colors='white')
    ax3.set_title('Volatility', color='white')
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_ylim(-2.1, 2.1)  # Adjust y-axis range to prevent clipping

    fig.canvas.draw_idle()

slider.on_changed(update)

# Initialize plot
update(0)

plt.show()