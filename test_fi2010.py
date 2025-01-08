import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import logging
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.ocet import OCET  # Ensure this import is correct for your project structure
import matplotlib.pyplot as plt
import gc

# Configure CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

def load_data(file_paths):
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
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    lob[lob == 0] = 2
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)

    dataX = np.zeros((N - T + 1, T, D), dtype=np.float32)
    dataY = dY[T - 1:N]
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

class LOBDataset(Dataset):
    def __init__(self, k, T, split):
        self.k = k
        self.T = T
        data_dir = 'data'

        if split == 'train':
            logging.info('Loading train data...')
            files = [os.path.join(data_dir, 'Train_Dst_NoAuction_DecPre_CF_7.txt')]
        elif split == 'val':
            logging.info('Loading validation data...')
            files = [
                os.path.join(data_dir, 'Test_Dst_NoAuction_DecPre_CF_7.txt'),
                os.path.join(data_dir, 'Test_Dst_NoAuction_DecPre_CF_8.txt')
            ]
        elif split == 'test':
            logging.info('Loading test data...')
            files = [
                os.path.join(data_dir, 'Test_Dst_NoAuction_DecPre_CF_9.txt')
            ]
        else:
            raise ValueError("Split must be either 'train', 'val', or 'test'")

        data = load_data(files)
        lob = prepare_x(data)
        label = get_label(data)
        trainX_CNN, trainY_CNN = data_classification(lob, label, self.T)
        trainY_CNN = trainY_CNN[:, self.k] - 1

        self.lob = torch.from_numpy(trainX_CNN).permute(0, 3, 1, 2).float()
        self.label = torch.from_numpy(trainY_CNN).long()

        logging.info(f'{split} data shape: {self.lob.shape}')
        logging.info(f'{split} label shape: {self.label.shape}')
        logging.info(f'Label distribution: {torch.bincount(self.label)}')

    def __getitem__(self, index):
        return self.lob[index, :, :, :], self.label[index]

    def __len__(self):
        return self.lob.size(0)

def train(epoch, model, dataloader_train, dataloader_val, optimizer, loss_fn, scheduler, epochs):
    start_time = time.time()
    model.train()
    train_correct, train_total = 0.0, 0.0
    train_loss_epoch = []

    pbar = tqdm(total=len(dataloader_train), desc=f"Epoch {epoch}/{epochs}", ncols=100, leave=False, position=0)

    for i, (lob, label) in enumerate(dataloader_train):
        lob, label = lob.cuda(), label.cuda()
        optimizer.zero_grad()
        pred = model(lob)
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(pred.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()
        train_loss_epoch.append(loss.item())

        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{train_correct / train_total:.4f}"})
        pbar.update(1)

    pbar.close()

    train_acc = train_correct / train_total
    train_loss = np.mean(train_loss_epoch)

    val_acc, val_loss = evaluate(model, dataloader_val, loss_fn, epoch, "Validation")

    end_time = time.time()
    logging.info("Epoch %d/%s | Train Loss: %f | Train Acc: %f | Val Loss: %f | Val Acc: %f | Time: %f s",
                 epoch, epochs, train_loss, train_acc, val_loss, val_acc, end_time - start_time)

    scheduler.step(val_acc)

    return train_acc, train_loss, val_acc, val_loss

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, epoch, split="Test"):
    model.eval()
    correct = 0.0
    total = 0.0
    loss_epoch = []
    for (lob, label) in tqdm(dataloader, desc=f"Evaluating {split}", ncols=100):
        lob, label = lob.cuda(), label.cuda()
        outputs = model(lob)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += float(torch.sum(predicted == label))

        loss = loss_fn(outputs, label)
        loss_epoch.append(loss.item())

    acc = correct / total
    avg_loss = np.mean(loss_epoch)
    logging.info(f'{split} set: Epoch: {epoch}, Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}')
    return acc, avg_loss

def plot_training_process(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_process.png')
    plt.close()

def main():
    k = 4
    T = 100

    dataset_train = LOBDataset(k, T=T, split='train')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    dataset_val = LOBDataset(k, T=T, split='val')
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    dataset_test = LOBDataset(k, T=T, split='test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=256, shuffle=False)

    model_name = 'ocet'
    save_k = ['k_10', 'k_20', 'k_30', 'k_50', 'k_100']
    save_path = f'model_save/FI2010/{save_k[k]}/{model_name}/'
    os.makedirs(save_path, exist_ok=True)

    model = OCET(
        num_classes=3,
        dim=100,
        depth=5,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        dropout=0.2
    ).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    logging.info('Model = %s', str(model))
    logging.info('Model parameters = %d', sum(p.numel() for p in model.parameters()))
    logging.info('Train num = %d', len(dataset_train))
    logging.info('Val num = %d', len(dataset_val))
    logging.info('Test num = %d', len(dataset_test))

    epochs = 50
    best_val_acc = 0.0
    best_epoch = 1
    best_loss = float('inf')
    best_model_path = None
    patience = 8  # 连续5个epoch没有改善才考虑加载最佳模型
    no_improvement_count = 0
    improvement_threshold = 0.001  # 0.1%的改善被视为有效改善

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        logging.info('Start training epoch %d...', epoch)

        train_acc, train_loss, val_acc, val_loss = train(epoch, model, dataloader_train, dataloader_val, optimizer, loss_fn, scheduler, epochs)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_acc > best_val_acc + improvement_threshold:
            best_val_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch
            best_model_path = f"{save_path}best_model_Acc{best_val_acc:.4f}_Los{best_loss:.2f}_E{best_epoch}.pth"
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
                    no_improvement_count = 0  # 重置计数器
                else:
                    logging.info(
                        f'No improvement for {patience} epochs, but no best model saved yet. Continuing with current model.')

    logging.info('Current best validation accuracy: %.4f (epoch %d) | Best loss: %.4f',
                     best_val_acc, best_epoch, best_loss)

    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load(best_model_path))
    test_acc, test_loss = evaluate(model, dataloader_test, loss_fn, epoch, "Test")
    logging.info(f'Final Test set: Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}')

    plot_training_process(train_losses, val_losses, train_accs, val_accs)

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()