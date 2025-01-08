import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

def check_label_distribution_txt(output_folder):
    """
    Checks the distribution of two specific labels (label_1 and label_10) across all TXT files in the specified folder
    and generates vertically stacked subplots showing the distribution from the 9th to the 20th day.

    Parameters:
    output_folder (str): Path to the folder containing the processed TXT files.
    """
    # Define the label names to be displayed
    label_names = ['label_1', 'label_10']

    # Initialize a nested dictionary to hold counts for each label category per day
    # Structure: {day: {label_name: {category: count}}}
    label_counts_per_day = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Define the date range of interest
    start_day = 9
    end_day = 20
    days_of_interest = list(range(start_day, end_day + 1))

    # Create a mapping for label categories
    category_mapping = {
        1: 'Upward Movement',
        2: 'No Significant Change',
        3: 'Downward Movement'
    }

    # Get all TXT files in the output folder
    file_pattern = os.path.join(output_folder, "*.txt")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"No TXT files found in the folder: {output_folder}")
        return

    print(f"Found {len(file_list)} TXT files in the folder: {output_folder}\n")

    for file_path in tqdm(file_list, desc="Processing files"):
        try:
            # Extract the day from the file name
            file_name = os.path.basename(file_path)
            # Assuming the file name format is 'BTC_DecPre_data_1_09.txt'
            day_str = file_name.split('_')[-1].split('.')[0]
            day = int(day_str)

            # Only process files within the specified date range
            if day not in days_of_interest:
                continue

            # Load the data from the TXT file
            data = np.loadtxt(file_path, delimiter=' ')

            if data.ndim != 2:
                print(f"Unexpected data format in file: {file_path}")
                continue

            num_rows, num_cols = data.shape

            if num_rows < len(label_names):
                print(f"Not enough rows in file: {file_path}")
                continue

            # Extract the last rows corresponding to the selected labels
            label_data = data[-len(label_names):, :]  # Shape: (2, num_samples)

            for i, label in enumerate(label_names):
                # Extract the current label row
                current_label_row = label_data[i, :]

                # Convert float labels to integers
                current_labels = current_label_row.astype(int)

                # Count occurrences of each category
                unique, counts = np.unique(current_labels, return_counts=True)
                for u, c in zip(unique, counts):
                    category = category_mapping.get(u, f"Unknown({u})")
                    label_counts_per_day[day][label][category] += c

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # Prepare data for plotting
    distribution_records = []
    for day in sorted(label_counts_per_day.keys()):
        for label in label_names:
            for category, count in label_counts_per_day[day][label].items():
                distribution_records.append({
                    'Day': day,
                    'Label': label,
                    'Category': category,
                    'Count': count
                })

    df_distribution = pd.DataFrame(distribution_records)

    if df_distribution.empty:
        print("No label data found within the specified date range.")
        return

    # Set the plotting style
    sns.set(style="whitegrid")

    # Initialize the plot with vertically stacked subplots
    num_labels = len(label_names)
    fig, axes = plt.subplots(num_labels, 1, figsize=(20, 4 * num_labels), sharex=True, constrained_layout=True)

    # Adjust the space between subplots to prevent overlap
    fig.subplots_adjust(hspace=0.4)

    if num_labels == 1:
        axes = [axes]  # Ensure axes is iterable

    # Define color palette (consistent across subplots)
    palette = sns.color_palette("tab10", n_colors=len(category_mapping))

    for ax, label in zip(axes, label_names):
        # Filter data for the current label
        label_data = df_distribution[df_distribution['Label'] == label]

        # Pivot the data to have categories as columns
        pivot_df = label_data.pivot(index='Day', columns='Category', values='Count').fillna(0)

        # Ensure all categories are present
        for category in category_mapping.values():
            if category not in pivot_df.columns:
                pivot_df[category] = 0

        # Sort the columns based on the defined category order
        pivot_df = pivot_df[sorted(category_mapping.values(), key=lambda x: list(category_mapping.values()).index(x))]

        # Plot each category as a separate line
        for category, color in zip(category_mapping.values(), palette):
            if category in pivot_df.columns:
                sns.lineplot(
                    x=pivot_df.index,
                    y=pivot_df[category],
                    marker='o',
                    label=category,
                    ax=ax,
                    color=color
                )
                # Annotate each data point with its count
                for x, y in zip(pivot_df.index, pivot_df[category]):
                    if category == 'Downward Movement':
                        va = 'top'  # Place text below the point
                        offset = -25  # Adjust vertical offset
                    else:
                        va = 'bottom'  # Place text above the point
                        offset = 25  # Adjust vertical offset
                    ax.text(
                        x,
                        y,
                        int(y),
                        ha='center',
                        va=va,
                        fontsize=12,
                        color=color,
                        fontweight='bold'
                    )

        # Customize each subplot
        ax.set_title(f'Distribution of {label}', fontsize=18, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=14)
        ax.legend(title='Category', fontsize=12, title_fontsize=14)

    # Add a super title for the entire figure
    fig.suptitle('Label Distribution from Day 9 to Day 20', fontsize=22, fontweight='bold', y=1.02)

    # Customize the shared x-axis
    axes[-1].set_xlabel('Day of the Month', fontsize=16)
    axes[-1].set_xticks(days_of_interest)  # Ensure all days are shown on x-axis
    axes[-1].tick_params(axis='x', labelsize=12)

    # Adjust layout to prevent overlapping and make room for the super title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Print detailed label distribution
    print("\nDetailed Label Distribution Statistics:\n")
    for day in sorted(label_counts_per_day.keys()):
        print(f"Day {day}:")
        for label in label_names:
            print(f"  {label}:")
            for category, count in label_counts_per_day[day][label].items():
                print(f"    {category}: {count} samples")
        print()

# Example usage
if __name__ == "__main__":
    output_folder = "../data/processed"  # Replace with your actual output folder path containing TXT files
    check_label_distribution_txt(output_folder)
