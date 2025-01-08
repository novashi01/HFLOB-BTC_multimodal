import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def read_and_preprocess_csv(file_path):
    """
    Read and preprocess CSV file containing trade and order book data.
    
    Args:
        file_path (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional features
        
    Raises:
        ValueError: If required columns are missing in the input file
    """
    # Load CSV file
    df = pd.read_csv(file_path)
    print(f"File loaded successfully: {file_path}")
    print(f"Data shape: {df.shape}")

    # Check if required columns exist
    required_columns = ['timestamp', 'price', 'quantity', 'is_buyer_maker']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"Missing required columns: {set(required_columns) - set(df.columns)}")

    # Convert 'timestamp' to datetime format
    if 'datetime' in df.columns:
        df['time'] = pd.to_datetime(df['datetime'])
    else:
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Remove rows where 'price' is 0
    initial_shape = df.shape
    df = df[df['price'] != 0]
    print(f"Removed rows with 'price' = 0: {initial_shape[0] - df.shape[0]} rows deleted")
    print(f"Updated data shape: {df.shape}")

    # Optional: Remove rows where 'quantity' is 0
    # initial_shape = df.shape
    # df = df[df['quantity'] != 0]
    # print(f"Removed rows with 'quantity' = 0: {initial_shape[0] - df.shape[0]} rows deleted")
    # print(f"Updated data shape: {df.shape}")

    # Fill 0 values in 'quantity' with previous non-zero values
    df['quantity'].replace(0, np.nan, inplace=True)
    df['quantity'].ffill(inplace=True)

    # Check if 'quantity' still contains 0 values after filling
    if (df['quantity'] == 0).any():
        print("Warning: 'quantity' column still contains 0 values after filling.")

    # Calculate weighted mid-price using first two levels of LOB data
    df = calculate_weighted_mid_price(df)

    # Calculate relative mid-price using aggregated trade data
    df = calculate_relative_mid_price(df)

    # Check if 'mid_price' and 'relative_mid_price' are calculated correctly
    if df['mid_price'].isnull().any():
        print("Warning: 'mid_price' contains NaN values.")
    if df['relative_mid_price'].isnull().any():
        print("Warning: 'relative_mid_price' contains NaN values.")

    # Extract LOB data columns
    ask_price_cols = [col for col in df.columns if col.startswith('ask') and not col.startswith('ask_vol')]
    ask_volume_cols = [col for col in df.columns if col.startswith('ask_vol')]
    bid_price_cols = [col for col in df.columns if col.startswith('bid') and not col.startswith('bid_vol')]
    bid_volume_cols = [col for col in df.columns if col.startswith('bid_vol')]

    # Check if LOB data exists
    if not ask_price_cols or not bid_price_cols:
        print("LOB data not found in CSV file.")
    else:
        # Normalize LOB data using decimal scaling
        df = normalize_lob_data(df, ask_price_cols + bid_price_cols + ask_volume_cols + bid_volume_cols)

    # Add new features: vwap, order_book_imbalance, trade_flow
    df = add_new_features(df)

    # Add time-based features: sin_time and cos_time
    df = add_time_sin_cos_features(df)

    return df


def calculate_weighted_mid_price(df, depth=2):
    """
    Calculate weighted mid-price using LOB data.

    Args:
        df (pd.DataFrame): DataFrame containing order book data
        depth (int): Number of price levels to consider, default is 2, max is 10

    Returns:
        pd.DataFrame: DataFrame with added 'mid_price' column
        
    Raises:
        ValueError: If required LOB columns are missing
    """
    # Limit depth to maximum of 10 levels
    depth = min(depth, 10)

    # Generate required LOB column names based on depth
    required_lob_columns = []
    for i in range(1, depth + 1):
        required_lob_columns.extend([
            f'ask0{i}', f'ask_vol0{i}',
            f'bid0{i}', f'bid_vol0{i}'
        ])
    print(required_lob_columns)

    # Check for required LOB columns
    missing_columns = [col for col in required_lob_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required LOB columns: {missing_columns}")

    # Extract prices and volumes for each level
    ask_prices = np.array([df[f"ask0{i}"] for i in range(1, depth + 1)])
    ask_volumes = np.array([df[f"ask_vol0{i}"] for i in range(1, depth + 1)])
    bid_prices = np.array([df[f"bid0{i}"] for i in range(1, depth + 1)])
    bid_volumes = np.array([df[f"bid_vol0{i}"] for i in range(1, depth + 1)])

    # Calculate weighted ask and bid prices
    ask_price_volume_sum = np.sum(ask_prices * ask_volumes, axis=0)
    ask_volume_sum = np.sum(ask_volumes, axis=0)
    weighted_ask_price = ask_price_volume_sum / ask_volume_sum

    bid_price_volume_sum = np.sum(bid_prices * bid_volumes, axis=0)
    bid_volume_sum = np.sum(bid_volumes, axis=0)
    weighted_bid_price = bid_price_volume_sum / bid_volume_sum

    # Calculate mid-price
    mid_price = (weighted_ask_price + weighted_bid_price) / 2
    df['mid_price'] = mid_price.round(6)

    # Check for NaN values and forward fill if necessary
    if df['mid_price'].isnull().any():
        print("Warning: NaN values found in weighted mid-price calculation, using forward fill.")
        df['mid_price'].ffill(inplace=True)

    return df


def calculate_relative_mid_price(df):
    """
    Calculate relative mid-price using trade data.

    Args:
        df (pd.DataFrame): DataFrame containing trade data

    Returns:
        pd.DataFrame: DataFrame with added 'relative_mid_price' column
        
    Raises:
        ValueError: If required trade columns are missing
    """
    # Check for required trade data columns
    required_trade_columns = ['price', 'quantity']
    missing_columns = [col for col in required_trade_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required trade columns: {missing_columns}")

    # Calculate relative mid-price using trade price directly
    df['relative_mid_price'] = df['price']

    # Round to 6 decimal places
    df['relative_mid_price'] = df['relative_mid_price'].round(6)

    # Handle NaN values
    if df['relative_mid_price'].isnull().any():
        print("Warning: NaN values found in relative mid-price calculation, using forward fill.")
        df['relative_mid_price'].ffill(inplace=True)

    # Fill remaining NaN values with global mean if any exist
    if df['relative_mid_price'].isnull().any():
        print("Warning: 'relative_mid_price' still contains NaN values, using global mean.")
        global_mean = df['relative_mid_price'].mean()
        df['relative_mid_price'].fillna(global_mean, inplace=True)

    return df


def normalize_lob_data(df, lob_columns, decimal_places=8):
    """
    Normalize LOB data using decimal scaling.

    Args:
        df (pd.DataFrame): DataFrame containing LOB data
        lob_columns (list): List of LOB column names to normalize
        decimal_places (int): Number of decimal places to round to

    Returns:
        pd.DataFrame: DataFrame with normalized LOB columns
    """
    for col in lob_columns:
        max_val = df[col].max()

        # Ensure max_val is positive for scaling factor calculation
        if pd.isna(max_val) or max_val <= 0:
            scaling_factor = 1
        else:
            scaling_factor = 10 ** len(str(int(max_val)))

        df[col] = (df[col] / scaling_factor).round(decimal_places)

    return df


def add_new_features(df, vwap_window=10, trade_flow_window=10):
    """
    Add new features: VWAP, order book imbalance, and trade flow.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        vwap_window (int): Window size for VWAP calculation
        trade_flow_window (int): Window size for trade flow calculation

    Returns:
        pd.DataFrame: DataFrame with added feature columns
    """
    # Calculate VWAP (Volume Weighted Average Price)
    df['vwap'] = (df['price'] * df['quantity']).rolling(window=vwap_window, min_periods=1).sum() / df['quantity'].rolling(window=vwap_window, min_periods=1).sum()

    # Calculate order book imbalance using first two levels
    df['total_bid_volume'] = df[['bid01', 'bid02']].sum(axis=1)
    df['total_ask_volume'] = df[['ask01', 'ask02']].sum(axis=1)
    df['order_book_imbalance'] = (df['total_bid_volume'] - df['total_ask_volume']) / (df['total_bid_volume'] + df['total_ask_volume'])
    
    # Replace inf/NaN values in order book imbalance
    df['order_book_imbalance'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['order_book_imbalance'].fillna(0, inplace=True)

    # Calculate trade flow
    # trade_flow = (2 * is_buyer_maker - 1) represents buy=1, sell=-1
    df['trade_direction'] = 2 * df['is_buyer_maker'] - 1
    df['trade_flow'] = df['trade_direction'].rolling(window=trade_flow_window, min_periods=1).sum()
    
    # Remove temporary columns
    df.drop(columns=['trade_direction', 'total_bid_volume', 'total_ask_volume'], inplace=True)

    # Handle NaN values
    df['vwap'].ffill(inplace=True)
    df['vwap'].bfill(inplace=True)
    df['trade_flow'].fillna(0, inplace=True)

    # Round new features to 6 decimal places
    df['vwap'] = df['vwap'].round(6)
    df['order_book_imbalance'] = df['order_book_imbalance'].round(6)
    df['trade_flow'] = df['trade_flow'].round(6)

    return df


def calculate_volatility_phase(df, window=100):
    """
    Calculate volatility and label market phases from -2 to 2 based on volatility levels.

    Args:
        df (pd.DataFrame): Input DataFrame
        window (int): Rolling window size for volatility calculation

    Returns:
        pd.DataFrame: DataFrame with added 'market_phase' column
    """
    df = df.reset_index(drop=True)
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

    # Define volatility thresholds
    low_threshold = df['volatility'].quantile(0.2)
    high_threshold = df['volatility'].quantile(0.8)

    # Assign market phases
    df['market_phase'] = 0  # Normal volatility
    df.loc[df['volatility'] <= low_threshold, 'market_phase'] = -2  # Very low volatility
    df.loc[(df['volatility'] > low_threshold) & (
            df['volatility'] <= df['volatility'].median()), 'market_phase'] = -1  # Low volatility
    df.loc[(df['volatility'] > df['volatility'].median()) & (
            df['volatility'] <= high_threshold), 'market_phase'] = 1  # High volatility
    df.loc[df['volatility'] > high_threshold, 'market_phase'] = 2  # Very high volatility

    # Remove temporary columns
    df.drop(['returns', 'volatility'], axis=1, inplace=True)
    return df


def generate_labels(df, k_values=[1, 2, 3, 5, 10], threshold=0.000005):
    """
    Generate labels based on future price movements.

    Args:
        df (pd.DataFrame): Input DataFrame with 'price' column
        k_values (list): List of future time steps to consider
        threshold (float): Price change threshold for label generation

    Returns:
        pd.DataFrame: DataFrame with added label columns
    """
    df = df.reset_index(drop=True)

    for k in k_values:
        column_name = f'label_{k}'

        # Get future prices k steps ahead
        future_price = df['price'].shift(-k)

        # Calculate percentage change
        percent_change = (future_price - df['price']) / df['price']

        # Initialize labels (2 = no significant change)
        df[column_name] = 2

        # Set labels based on threshold
        df.loc[percent_change >= threshold, column_name] = 1  # Price increase
        df.loc[percent_change <= -threshold, column_name] = 3  # Price decrease

        # Handle last k rows where future prices are unknown
        if len(df) > k:
            last_valid_label = df.loc[df.index[-k - 1], column_name]
            df.loc[df.index[-k:], column_name] = last_valid_label
        else:
            df[column_name] = 2

    return df


def time_to_sin_cos(time_series):
    """
    Convert time to sinusoidal features for capturing daily patterns.

    Args:
        time_series: datetime series

    Returns:
        pd.DataFrame: DataFrame with 'sin_time' and 'cos_time' columns
    """
    time_in_seconds = time_series.dt.hour * 3600 + time_series.dt.minute * 60 + time_series.dt.second
    max_seconds = 86400  # 24 hours in seconds
    time_in_seconds = time_in_seconds % max_seconds
    sin_time = np.sin(2 * np.pi * time_in_seconds / max_seconds)
    cos_time = np.cos(2 * np.pi * time_in_seconds / max_seconds)
    return pd.DataFrame({'sin_time': sin_time, 'cos_time': cos_time})


def add_time_sin_cos_features(df):
    """
    Add sinusoidal time features to capture daily cyclical patterns.

    Args:
        df (pd.DataFrame): DataFrame containing 'time' column in datetime format

    Returns:
        pd.DataFrame: DataFrame with added 'sin_time' and 'cos_time' columns
    """
    time_features = time_to_sin_cos(df['time'])
    df = pd.concat([df, time_features], axis=1)
    return df


def process_and_save_by_day(df, output_folder):
    """
    Process data by day and save to separate CSV files.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        output_folder (str): Path to output directory for saving processed files
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract date from datetime
    df['date'] = df['time'].dt.date
    grouped = df.groupby('date')

    for date, group in tqdm(grouped, desc="Processing data by day"):
        group = group.copy()
        
        # Calculate volatility phases
        group = calculate_volatility_phase(group)
        
        # Generate price movement labels
        group = generate_labels(group)
        
        # Select relevant columns including LOB data and new features
        lob_columns = [col for col in group.columns if col.startswith('ask') or col.startswith('bid')]
        label_columns = [f'label_{k}' for k in [1, 2, 3, 5, 10]]
        columns = ['time', 'sin_time', 'cos_time', 'mid_price', 'relative_mid_price', 
                  'vwap', 'order_book_imbalance', 'trade_flow', 'market_phase'] + lob_columns + label_columns
        group = group[columns]

        # Handle NaN values
        if group.isnull().values.any():
            print(f"Warning: NaN values found in data for date {date}")
            # Remove rows with NaN values
            group = group.dropna()
            # Alternative: forward fill NaN values
            # group.fillna(method='ffill', inplace=True)

        # Print label distribution statistics
        print(f"\nDate: {date}")
        print("Label Distribution:")
        print(group[label_columns].apply(pd.Series.value_counts))
        print("Market Phase Distribution:")
        print(group['market_phase'].value_counts())

        # Save to CSV file with 'mm_dd' format
        date_str = date.strftime('%m_%d')
        output_file = os.path.join(output_folder, f'processed_BTC_trade_{date_str}.csv')
        group.to_csv(output_file, index=False)
        print(f"File saved: {output_file}")


def main(input_file, output_folder):
    """
    Main function to execute the data processing pipeline.

    Args:
        input_file (str): Path to input CSV file
        output_folder (str): Path to output directory for processed files
    """
    df = read_and_preprocess_csv(input_file)
    process_and_save_by_day(df, output_folder)


if __name__ == "__main__":
    # Configuration parameters
    input_file = "../data/BTC_LOB_Trades_01_9-20.csv"
    output_folder = "../data/trade_lob_multi"
    main(input_file, output_folder)