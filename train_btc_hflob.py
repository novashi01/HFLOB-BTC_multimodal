import argparse
import gc
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import logging
import sys
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from model.ocet import OCET
from model.ocet import OCET2D
import matplotlib.pyplot as plt
import time
from datetime import datetime
from torch.nn import functional as F

# Configure CUDA and logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


def load_csv_data(data_path, start_day, end_day):
    """
    Load multiple CSV files containing trading data for a specified date range.
    
    Args:
        data_path (str): Directory path containing the CSV files
        start_day (int): First day to load (inclusive)
        end_day (int): Last day to load (inclusive)
    
    Returns:
        pd.DataFrame: Concatenated DataFrame containing all loaded data
        
    Raises:
        ValueError: If no data files were successfully loaded
    """
    all_data = []
    for day in range(start_day, end_day + 1):
        file_name = f'processed_BTC_trade_01_{day:02d}.csv'
        file_path = os.path.join(data_path, file_name)
        try:
            day_data = pd.read_csv(file_path)
            all_data.append(day_data)
            logging.info(f"Loaded data from {file_name}")
        except IOError as e:
            logging.error(f"Error loading {file_name}: {e}")

    if not all_data:
        raise ValueError("No data files were successfully loaded.")

    return pd.concat(all_data, ignore_index=True)


def prepare_x(data):
    """
    Extract limit order book (LOB) features from the data.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing LOB data
    
    Returns:
        np.ndarray: Array containing LOB features in the specified order
        
    Features are organized as follows:
    - First 20 columns: Ask prices and volumes (levels 1-10)
    - Last 20 columns: Bid prices and volumes (levels 1-10)
    """
    feature_columns = [
        'ask10', 'ask_vol10', 'ask09', 'ask_vol09', 'ask08', 'ask_vol08',
        'ask07', 'ask_vol07', 'ask06', 'ask_vol06', 'ask05', 'ask_vol05',
        'ask04', 'ask_vol04', 'ask03', 'ask_vol03', 'ask02', 'ask_vol02',
        'ask01', 'ask_vol01', 'bid01', 'bid_vol01', 'bid02', 'bid_vol02',
        'bid03', 'bid_vol03', 'bid04', 'bid_vol04', 'bid05', 'bid_vol05',
        'bid06', 'bid_vol06', 'bid07', 'bid_vol07', 'bid08', 'bid_vol08',
        'bid09', 'bid_vol09', 'bid10', 'bid_vol10'
    ]

    return data[feature_columns].values


def get_label(data, label_column='label_1'):
    """
    Transform labels for classification task.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing label column
        label_column (str): Name of the column containing labels
        
    Returns:
        np.ndarray: Transformed labels (0, 1, 2) representing price movement directions
    
    Note:
        - Original label 0 is mapped to 2
        - Labels are adjusted by -1 to start from 0
    """
    labels = data[label_column].values
    labels[labels == 0] = 2  # Map label 0 to 2
    return labels - 1  # Adjust labels to be 0-based (0, 1, 2)


def data_classification(X, Y, T):
    """
    Transform labels for classification task.
    
    Args:
        data (pd.DataFrame): Input DataFrame containing label column
        label_column (str): Name of the column containing labels
        
    Returns:
        np.ndarray: Transformed labels (0, 1, 2) representing price movement directions
    
    Note:
        - Original label 0 is mapped to 2
        - Labels are adjusted by -1 to start from 0
    """
    labels = data[label_column].values
    labels[labels == 0] = 2  # Map label 0 to 2
    return labels - 1  # Adjust labels to be 0-based (0, 1, 2)
    [N, D] = X.shape
    dataX = np.zeros((N - T + 1, T, D), dtype=np.float32)
    dataY = Y[T - 1:N]
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY


class LOBDataset(Dataset):
    """
    Custom Dataset class for Limit Order Book (LOB) data.
    
    Args:
        T (int): Sequence length for each sample
        split (str): Dataset split ('train', 'val', or 'test')
        scaler (StandardScaler, optional): Fitted scaler for feature normalization
        
    Features:
        - LOB data (prices and volumes)
        - Time features (sin and cos)
        - Market phase indicators
        - Additional market metrics (VWAP, trade flow, etc.)
    """
    def __init__(self, T, split, scaler=None):
        self.T = T
        data_path = 'data/trade_lob_multi6'
        if split == 'train':
            logging.info('Loading train data...')
            data = load_csv_data(data_path, 9, 16)
        elif split == 'val':
            logging.info('Loading validation data...')
            data = load_csv_data(data_path, 17, 18)
        elif split == 'test':
            logging.info('Loading test data...')
            data = load_csv_data(data_path, 19, 20)
        else:
            raise ValueError("Split must be either 'train', 'val', or 'test'")

        # Preprocess data to handle NaN values
        data = self.preprocess_data(data)

        lob = prepare_x(data)

        if scaler is None:
            self.scaler = StandardScaler()
            lob = self.scaler.fit_transform(lob)
        else:
            self.scaler = scaler
            lob = self.scaler.transform(lob)

        label = get_label(data)
        X_CNN, Y_CNN = data_classification(lob, label, self.T)

        self.lob = torch.from_numpy(X_CNN).permute(0, 3, 1, 2).float()

        # Handle sin and cos time
        self.sin_time = torch.from_numpy(data['sin_time'].values[self.T - 1:, None]).float()
        self.cos_time = torch.from_numpy(data['cos_time'].values[self.T - 1:, None]).float()

        # Additional features
        self.vwap = torch.from_numpy(data['vwap'].values[self.T - 1:, None]).float()
        self.trade_flow = torch.from_numpy(data['trade_flow'].values[self.T - 1:, None]).float()
        self.cumulative_trade_flow = torch.from_numpy(data['relative_mid_price'].values[self.T - 1:, None]).float()
        self.order_book_imbalance = torch.from_numpy(data['order_book_imbalance'].values[self.T - 1:, None]).float()

        # Convert market_phase to one-hot encoding
        market_phase_categories = pd.Categorical(data['market_phase'])
        market_phase_one_hot = pd.get_dummies(market_phase_categories)
        self.market_phase = torch.from_numpy(market_phase_one_hot.values[self.T - 1:]).float()

        self.label = torch.from_numpy(Y_CNN).long()
        # Add mid_price for backtesting
        self.mid_price = data['mid_price'].values[self.T - 1:].astype(np.float32)

        logging.info(f'{split} data shape: {self.lob.shape}')
        logging.info(f'{split} label shape: {self.label.shape}')
        logging.info(f'{split} market phase shape: {self.market_phase.shape}')
        logging.info(f'{split} label distribution: {np.bincount(Y_CNN)}')

    def preprocess_data(self, data):
        """
        Preprocess the dataset by handling missing values.
        
        Args:
            data (pd.DataFrame): Raw input DataFrame
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with no missing values
            
        Note:
            - Numeric columns: NaN values are replaced with column minimums
            - Non-numeric columns: NaN values are replaced with mode or 'unknown'
        """
        # Check for NaN values
        nan_columns = data.columns[data.isna().any()].tolist()
        if nan_columns:
            logging.warning(f"NaN values found in columns: {nan_columns}")

        # Replace NaN with minimum values for each column
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                min_value = data[col].min()
                data[col].fillna(min_value, inplace=True)
            else:
                # For non-numeric columns, use the mode
                if not data[col].mode().empty:
                    data[col].fillna(data[col].mode()[0], inplace=True)
                else:
                    # If mode is empty (all values are NaN), fill with a placeholder
                    data[col].fillna('unknown', inplace=True)

        # Log the number of NaN values replaced
        nan_count = data.isna().sum().sum()
        logging.info(f"Total NaN values replaced: {nan_count}")

        return data

    def __getitem__(self, index):
        return (self.lob[index], self.label[index], self.sin_time[index], self.cos_time[index],
                self.market_phase[index], self.vwap[index], self.trade_flow[index],
                self.cumulative_trade_flow[index], self.order_book_imbalance[index], self.mid_price[index])

    def __len__(self):
        return self.lob.size(0)


class ImprovedLOBModelWithMultiBranch(nn.Module):
    """
    Enhanced Limit Order Book Model with multiple feature branches.
    
    This model combines LOB data with various market features using a multi-branch
    architecture for improved price movement prediction.
    
    Args:
        num_classes (int): Number of output classes (typically 3 for price movement)
        dim (int): Base dimension for feature processing
        depth (int): Number of transformer layers
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        mlp_dim (int): Dimension of the MLP layer
        dropout (float): Dropout rate
        num_market_phases (int): Number of distinct market phases
        use_multimodal (bool): Whether to use additional market features
    """
    def __init__(self, num_classes, dim, depth, heads, dim_head, mlp_dim, dropout, num_market_phases, use_multimodal=True):
        super(ImprovedLOBModelWithMultiBranch, self).__init__()

        self.use_multimodal = use_multimodal

        self.lob_model = OCET(
            num_classes=dim,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        if self.use_multimodal:
            self.price_quantity_features = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, 64)
            )

            self.market_phase_features = nn.Sequential(
                nn.Linear(num_market_phases, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, 64)
            )

            self.additional_features = nn.Sequential(
                nn.Linear(4, 64),  # vwap, trade_flow, cumulative_trade_flow, order_book_imbalance
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
                nn.Linear(64, 64)
            )

            self.gru = nn.GRU(input_size=192, hidden_size=128, num_layers=3, batch_first=True, dropout=dropout)

            self.fusion = nn.Sequential(
                nn.Linear(dim + 128, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(dim // 2),
                nn.Dropout(dropout)
            )

            self.predictor = nn.Linear(dim // 2, num_classes)
        else:
            self.predictor = nn.Linear(dim, num_classes)

    def forward(self, lob, sin_time=None, cos_time=None, market_phase=None, vwap=None, trade_flow=None, cumulative_trade_flow=None, order_book_imbalance=None):
        lob_features = self.lob_model(lob)
        """
        Forward pass of the model.
        
        Args:
            lob (torch.Tensor): Limit order book data
            sin_time (torch.Tensor, optional): Sine of time feature
            cos_time (torch.Tensor, optional): Cosine of time feature
            market_phase (torch.Tensor, optional): Market phase indicators
            vwap (torch.Tensor, optional): Volume-weighted average price
            trade_flow (torch.Tensor, optional): Trade flow indicator
            cumulative_trade_flow (torch.Tensor, optional): Cumulative trade flow
            order_book_imbalance (torch.Tensor, optional): Order book imbalance
            
        Returns:
            torch.Tensor: Class probabilities for price movement prediction
        """
        # Process LOB data through main branch
        lob_features = self.lob_model(lob)

        if self.use_multimodal:
            # Process time features
            price_quantity = torch.cat([sin_time, cos_time], dim=-1)
            price_quantity_features = self.price_quantity_features(price_quantity)
            
            # Process market phase features
            market_phase_features = self.market_phase_features(market_phase)
            
            # Process additional market metrics
            additional_features = self.additional_features(
                torch.cat([vwap, trade_flow, cumulative_trade_flow, order_book_imbalance], dim=-1)
            )

            # Combine all feature branches
            combined_features = torch.cat([
                price_quantity_features,
                market_phase_features,
                additional_features
            ], dim=-1)

            # Process through GRU
            gru_output, _ = self.gru(combined_features.unsqueeze(1))
            gru_features = gru_output[:, -1, :]

            # Fuse features and make prediction
            fused_features = self.fusion(torch.cat([lob_features, gru_features], dim=1))
            return self.predictor(fused_features)
        else:
            # Direct prediction from LOB features
            return self.predictor(lob_features)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    This loss function reduces the relative loss for well-classified examples and
    puts more focus on hard, misclassified examples.
    
    Args:
        alpha (float): Weighting factor, default 1
        gamma (float): Focusing parameter, default 2
        reduction (str): Reduction method ('mean' or 'sum')
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def compute_metrics_sklearn(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return accuracy, f1


def train(epoch, model, dataloader_train, dataloader_val, optimizer, loss_fn, scheduler, epochs, use_multimodal):
    start_time = time.time()
    model.train()
    train_loss_epoch = []
    train_f1_epoch = []

    pbar = tqdm(total=len(dataloader_train), desc=f"Epoch {epoch}/{epochs}", ncols=100, leave=False, position=0)

    # Create progress bar for training
    pbar = tqdm(total=len(dataloader_train), desc=f"Epoch {epoch}/{epochs}", ncols=100, leave=False, position=0)

    for i, batch in enumerate(dataloader_train):
        # Unpack batch data based on model type
        if use_multimodal:
            lob, label, sin_time, cos_time, market_phase, vwap, trade_flow, cumulative_trade_flow, order_book_imbalance, mid_price = [
                x.cuda() for x in batch]
        else:
            lob, label, _, _, _, _, _, _, _, mid_price = batch
            lob, label = lob.cuda(), label.cuda()

        # Forward pass and loss calculation
        optimizer.zero_grad()
        pred = model(lob, sin_time, cos_time, market_phase, vwap, trade_flow, cumulative_trade_flow, order_book_imbalance) if use_multimodal else model(lob)
        loss = loss_fn(pred, label)

        # Backward pass and optimization
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate metrics
        _, f1 = compute_metrics_sklearn(pred.argmax(dim=1).cpu().numpy(), label.cpu().numpy())
        train_loss_epoch.append(loss.item())
        train_f1_epoch.append(f1)

        # Update progress bar
        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'F1': f"{f1:.4f}"})
        pbar.update(1)

    pbar.close()

    # Calculate average metrics
    train_loss = np.mean(train_loss_epoch)
    train_f1 = np.mean(train_f1_epoch)

    # Evaluate on validation set
    val_loss, val_accuracy, val_f1, val_predictions, val_labels, val_probs, val_mid_prices = evaluate(
        model, dataloader_val, loss_fn, epoch, "Validation", use_multimodal)

    # Log training results
    end_time = time.time()
    logging.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | "
                 f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | "
                 f"Time: {end_time - start_time:.2f} s")

    # Update learning rate scheduler
    scheduler.step(val_f1)

    return train_loss, train_f1, val_loss, val_accuracy, val_f1, val_predictions, val_labels


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, epoch, split="Test", use_multimodal=True):
    model.eval()
    loss_epoch = []
    all_predictions = []
    all_labels = []
    all_probs = []
    all_mid_prices = []

    for batch in tqdm(dataloader, desc=f"Evaluating {split}", ncols=100):
        if use_multimodal:
            lob, label, sin_time, cos_time, market_phase, vwap, trade_flow, cumulative_trade_flow, order_book_imbalance, mid_price = [
                x.cuda() for x in batch]
            outputs = model(lob, sin_time, cos_time, market_phase, vwap, trade_flow, cumulative_trade_flow, order_book_imbalance)
        else:
            lob, label, sin_time, cos_time, market_phase, vwap, trade_flow, cumulative_trade_flow, order_book_imbalance, mid_price = batch
            lob, label = lob.cuda(), label.cuda()
            outputs = model(lob)

        # Calculate loss and predictions
        loss = loss_fn(outputs, label)
        preds = outputs.argmax(dim=1).cpu().numpy()
        labels_np = label.cpu().numpy()
        probs = F.softmax(outputs, dim=1).cpu().numpy()

        # Collect metrics
        loss_epoch.append(loss.item())
        all_predictions.extend(preds)
        all_labels.extend(labels_np)
        all_probs.extend(probs)
        all_mid_prices.extend(mid_price.cpu().numpy())  

    # Calculate average metrics
    avg_loss = np.mean(loss_epoch)
    avg_accuracy = accuracy_score(all_labels, all_predictions)
    avg_f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Log results
    logging.info(
        f'{split} set: Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, F1 Score: {avg_f1:.4f}')

    return avg_loss, avg_accuracy, avg_f1, np.array(all_predictions), np.array(all_labels), np.array(all_probs), np.array(all_mid_prices)


def plot_training_process(train_losses, val_losses, train_f1s, val_f1s, test_loss, test_f1, model_name):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(bottom=0)

    # Plot F1 score curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, val_f1s, label='Validation F1')
    plt.axhline(y=test_f1, color='r', linestyle='--', label='Test F1')
    plt.title('Training, Validation, and Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.ylim(0, 1)

    # Save plot
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    filename = f'{timestamp}_{model_name}_training_process.png'
    plt.savefig(filename)
    plt.close()

    logging.info(f'Training process plot saved as {filename}')


def backtest(predictions, labels, probs, price_data):
    """
    Perform backtesting on the model's predictions.
    
    Args:
        predictions (np.array): Model predictions
        labels (np.array): True labels
        probs (np.array): Prediction probabilities
        price_data (np.array): Mid prices for return calculation
        
    Calculates and prints:
        - Classification metrics (Accuracy, F1 Score)
        - Trading performance metrics (Total Return, Sharpe Ratio, Max Drawdown)
        - Saves a cumulative returns plot
    """
    # Calculate returns based on predictions
    returns = []
    for i in range(len(predictions) - 1):
        price_change = (price_data[i + 1] - price_data[i]) / price_data[i]

        if predictions[i] == 0:  # Predicted price increase
            returns.append(price_change)
        elif predictions[i] == 2:  # Predicted price decrease
            returns.append(-price_change)
        else:  # No significant change predicted
            returns.append(0)

    returns = np.array(returns)
    cumulative_returns = np.cumsum(returns)

    # Calculate Sharpe ratio (annualized)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

    # Calculate maximum drawdown
    peak = cumulative_returns[0]
    max_drawdown = 0
    for return_ in cumulative_returns[1:]:
        if return_ > peak:
            peak = return_
        drawdown = (peak - return_) / peak
        max_drawdown = max(max_drawdown, drawdown)

    # Print performance metrics
    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(labels, predictions, average='weighted'):.4f}")
    print(f"Total Return: {cumulative_returns[-1]:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns)
    plt.title("Cumulative Returns")
    plt.xlabel("Trade")
    plt.ylabel("Cumulative Return")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"cumulative_returns_{timestamp}.png")
    plt.close()

def main():
    """
    Main function to run the training pipeline.
    
    Handles:
        - Argument parsing
        - Data loading and preparation
        - Model creation and training
        - Model evaluation and backtesting
        - Results visualization and saving
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train LOB model with or without multimodal features")
    parser.add_argument("--use_multimodal", action="store_true", help="Use multimodal features")
    args = parser.parse_args()

    # Set sequence length and prepare datasets
    T = 100
    dataset_train = LOBDataset(T=T, split='train')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=512, shuffle=True)
    dataset_val = LOBDataset(T=T, split='val', scaler=dataset_train.scaler)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=512, shuffle=False)
    dataset_test = LOBDataset(T=T, split='test', scaler=dataset_train.scaler)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=512, shuffle=False)

    # Get number of market phases for model configuration
    num_market_phases = dataset_train.market_phase.shape[1]

    # Setup model saving directory
    model_name = 'improved_lob_model'
    save_path = f'model_save/BTC/{model_name}/'
    os.makedirs(save_path, exist_ok=True)

    # Initialize model and training parameters
    epochs = 50
    model = ImprovedLOBModelWithMultiBranch(
        num_classes=3,
        dim=100,
        depth=3,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        dropout=0.2,
        num_market_phases=num_market_phases,
        use_multimodal=args.use_multimodal
    ).cuda()

    logging.info(model)

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    loss_fn = FocalLoss(alpha=0.25, gamma=2)

    logging.info('Model = %s', str(model))
    logging.info('Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('Train num = %d', len(dataset_train))
    logging.info('Val num = %d', len(dataset_val))
    logging.info('Test num = %d', len(dataset_test))
    logging.info(f'Using multimodal features: {args.use_multimodal}')

    # Initialize training tracking variables
    best_val_f1 = 0.0
    best_epoch = 1
    best_model_path = None
    no_improvement_count = 0
    improvement_threshold = 0.0001
    patience = 8
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    # Training loop
    for epoch in range(1, epochs + 1):
        logging.info(f'Start training epoch {epoch}...')

        train_loss, train_f1, val_loss, val_accuracy, val_f1, val_predictions, val_labels = train(
            epoch, model, dataloader_train, dataloader_val, optimizer, loss_fn, scheduler, epochs, args.use_multimodal)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if val_f1 > best_val_f1 + improvement_threshold:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_path = f"{save_path}best_model_F1{best_val_f1:.4f}_E{best_epoch}.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved to {best_model_path}')
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                if best_model_path:
                    model.load_state_dict(torch.load(best_model_path))
                    logging.info(
                        f'No improvement for {patience} epochs. Loaded best model from {best_model_path} for continued training')
                    no_improvement_count = 0
                else:
                    logging.info(
                        f'No improvement for {patience} epochs, but no best model saved yet. Continuing with current model.')

    logging.info(f'Current best validation F1 score: {best_val_f1:.4f} (epoch {best_epoch})')

    # Load the best model and evaluate it on the test set
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
    else:
        logging.warning("No best model was saved during training.")

    # Evaluate the model on the test set
    test_loss, test_accuracy, test_f1, test_predictions, test_labels, test_probs, test_mid_prices = evaluate(
        model, dataloader_test, loss_fn, best_epoch, "Test", args.use_multimodal)

    logging.info(f'Final Test set: Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Loss: {test_loss:.4f}')

    plot_training_process(train_losses, val_losses, train_f1s, val_f1s, test_loss, test_f1, model_name)

    # Print the classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))

    # Perform backtesting
    backtest(test_predictions, test_labels, test_probs, test_mid_prices)  

    # Clear the GPU cache and perform garbage collection
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
