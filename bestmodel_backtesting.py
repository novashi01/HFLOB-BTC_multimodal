import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import sys
from model.ocet import OCET  # Import the OCET model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn import functional as F
from datetime import datetime
import matplotlib.pyplot as plt

# Configure CUDA settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def load_data(file_paths):
    """
    Load and concatenate data from multiple files.
    
    Args:
        file_paths (list): List of paths to data files
        
    Returns:
        numpy.ndarray: Concatenated data from all files
    """
    all_data = []
    for file_path in file_paths:
        try:
            day_data = np.loadtxt(file_path)
            all_data.append(day_data)
            logging.info(f"Loaded data from {file_path}")
        except IOError as e:
            logging.error(f"Error loading {file_path}: {e}")

    if not all_data:
        raise ValueError("No data files were successfully loaded.")

    return np.hstack(all_data)

def prepare_x(data):
    """
    Prepare input features from raw data.
    
    Args:
        data (numpy.ndarray): Raw input data
        
    Returns:
        numpy.ndarray: Processed input features
    """
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    """
    Extract and process labels from raw data.
    
    Args:
        data (numpy.ndarray): Raw input data
        
    Returns:
        numpy.ndarray: Processed labels
    """
    lob = data[-5:, :].T
    lob[lob == 0] = 2
    return lob

def data_classification(X, Y, T):
    """
    Prepare data for sequence classification.
    
    Args:
        X (numpy.ndarray): Input features
        Y (numpy.ndarray): Labels
        T (int): Sequence length
        
    Returns:
        tuple: Processed features and labels
    """
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)

    dataX = np.zeros((N - T + 1, T, D), dtype=np.float32)
    dataY = dY[T - 1:N]
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

class LOBDataset(Dataset):
    """
    Dataset class for Limit Order Book (LOB) data.
    
    Args:
        k (int): Label index to use
        T (int): Sequence length
        split (str): Dataset split ('test' only in this script)
    """
    def __init__(self, k, T, split):
        self.k = k
        self.T = T
        data_dir = 'data/processed7'  # Path to processed data directory

        if split == 'test':
            logging.info('Loading test data...')
            files = [os.path.join(data_dir, f'BTC_DecPre_data_1_{i}.txt') for i in range(19, 21)]
        else:
            raise ValueError("This dataset is only for 'test' split in this script.")

        data = load_data(files)
        lob = prepare_x(data)
        label = get_label(data)
        testX_CNN, testY_CNN = data_classification(lob, label, self.T)
        testY_CNN = testY_CNN[:, self.k] - 1

        self.lob = torch.from_numpy(testX_CNN).permute(0, 3, 1, 2).float()
        self.label = torch.from_numpy(testY_CNN).long()

        logging.info(f'{split} data shape: {self.lob.shape}')
        logging.info(f'{split} label shape: {self.label.shape}')
        logging.info(f'Label distribution: {torch.bincount(self.label)}')

    def __getitem__(self, index):
        return self.lob[index, :, :, :], self.label[index]

    def __len__(self):
        return self.lob.size(0)

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Args:
        alpha (float): Weighting factor
        gamma (float): Focusing parameter
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

@torch.no_grad()
def evaluate(model, dataloader, loss_fn):
    """
    Evaluate model on the given dataloader.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Test dataloader
        loss_fn (nn.Module): Loss function
        
    Returns:
        tuple: Loss, accuracy, F1 score, predictions, labels, and probabilities
    """
    model.eval()
    correct = 0.0
    total = 0.0
    loss_epoch = []
    all_predictions = []
    all_labels = []
    all_probs = []

    for (lob, label) in dataloader:
        lob, label = lob.cuda(), label.cuda()
        outputs = model(lob)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += float(torch.sum(predicted == label))

        loss = loss_fn(outputs, label)
        loss_epoch.append(loss.item())

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    acc = correct / total
    avg_loss = np.mean(loss_epoch)
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    logging.info(f'Test set: Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Loss: {avg_loss:.4f}')
    return avg_loss, acc, f1, np.array(all_predictions), np.array(all_labels), np.array(all_probs)

def load_price_data(T):
    """
    Load and process price data for backtesting.
    
    Args:
        T (int): Sequence length
        
    Returns:
        numpy.ndarray: Mid-prices aligned with predictions
    """
    data_dir = 'data/processed7'
    test_files = [os.path.join(data_dir, f'BTC_DecPre_data_1_{i}.txt') for i in range(19, 21)]
    data = load_data(test_files)

    # Extract unnormalized bid and ask prices
    ask_price = data[0, :]  # First row is ask price
    bid_price = data[19, :]  # Row 19 is bid price

    # Calculate mid-price
    mid_prices = (ask_price + bid_price) / 2

    # Align price data with predictions
    mid_prices = mid_prices[T - 1:]
    return mid_prices

def backtest(predictions, labels, probs, price_data, k):
    """
    Perform backtesting on model predictions.
    
    Args:
        predictions (numpy.ndarray): Model predictions
        labels (numpy.ndarray): True labels
        probs (numpy.ndarray): Prediction probabilities
        price_data (numpy.ndarray): Mid-prices
        k (int): Prediction horizon
    """
    correct_predictions = predictions == labels
    price_data = np.array(price_data)

    # Calculate returns based on price changes
    returns = []
    for i in range(len(predictions)):
        current_time = i
        future_time = i + k

        if future_time >= len(price_data):
            break

        price_change = (price_data[future_time] - price_data[current_time]) / price_data[current_time]

        if predictions[i] == 0:  # Predicted price decrease
            returns.append(-price_change)  # Short position return
        elif predictions[i] == 2:  # Predicted price increase
            returns.append(price_change)   # Long position return
        else:  # Predicted no change - no trade
            returns.append(0)

    # Truncate predictions and labels to match returns length
    predictions = predictions[:len(returns)]
    labels = labels[:len(returns)]
    correct_predictions = correct_predictions[:len(returns)]

    returns = np.array(returns)
    cumulative_returns = np.cumsum(returns)

    # Calculate Sharpe ratio
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

    # Calculate maximum drawdown
    cumulative_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_max - cumulative_returns) / cumulative_max
    max_drawdown = np.max(drawdown)

    # Log performance metrics
    logging.info(f"Backtesting Results:")
    logging.info(f"Accuracy: {np.mean(correct_predictions):.4f}")
    logging.info(f"F1 Score: {f1_score(labels, predictions, average='weighted'):.4f}")
    logging.info(f"Total Return: {cumulative_returns[-1]:.4f}")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    logging.info(f"Max Drawdown: {max_drawdown:.4f}")

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
    """Main function for model evaluation and backtesting."""
    k = 3  # Prediction horizon
    T = 100  # Sequence length

    # Load test dataset
    dataset_test = LOBDataset(k, T=T, split='test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=256, shuffle=False)

    # Initialize model architecture
    model = OCET(
        num_classes=3,
        dim=100,
        depth=3,
        heads=8,
        dim_head=25,
        mlp_dim=200,
        dropout=0.2
    ).cuda()

    # Load best model weights
    best_model_path = 'model_save/BTC/k_10/ocet/best_model_F10.8004_E15.pth'
    model.load_state_dict(torch.load(best_model_path))
    print(f"Model loaded from: {best_model_path}")

    # Define loss function
    loss_fn = FocalLoss(alpha=0.25, gamma=2)

    # Evaluate model on test set
    test_loss, test_accuracy, test_f1, test_predictions, test_labels, test_probs = evaluate(
        model, dataloader_test, loss_fn)
    logging.info(f'Final Test set: Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Loss: {test_loss:.4f}')

    # Output classification metrics
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))

    # Load price data and perform backtesting
    mid_prices = load_price_data(T)
    mid_prices = mid_prices[:len(test_predictions) + 1]
    backtest(test_predictions, test_labels, test_probs, mid_prices, k)

if __name__ == '__main__':
    main()