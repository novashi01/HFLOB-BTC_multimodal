# Multimodal Deep Learning for BTC Limit Order Book Data

## Overview
This repository contains the implementation of the **Hybrid Fusion with Limit Order Book (HFLOB)** model, as described in the research paper *"Adaptive Financial Forecasting with Multimodal Learning: Insights from Limit Order Book Data"* by Zhong Shi. The HFLOB model leverages advanced multimodal deep learning techniques to improve financial market prediction accuracy and profitability under complex market conditions.

## Features
- **Label Visualization for FI-2010**: Provides tools to visually confirm the labeling process of the FI-2010 dataset, ensuring the consistency and accuracy of mid-price changes.
- **Exponential Moving Average (EMA) for BTC Labeling**: Implements EMA-based trend detection for labeling BTC perpetual contract data, enabling comparative analysis between BTC and FI-2010 datasets.
- **Profitability Trade-Off Analysis**: Demonstrates that while the EMA labeling technique improves prediction accuracy for BTC, it results in a reduction in backtesting profitability metrics such as the Sharpe ratio.
- **Multimodal Data Integration**: Combines limit order book (LOB) data, technical indicators, and market sentiment for enhanced prediction.
- **State-of-the-Art Deep Learning Models**: Implements CNN, Transformer, and hybrid architectures for temporal and spatial feature extraction.
- **OCET Module Integration**: Utilizes the OCET (One-Dimensional Convolution Embedding Transformer) module for capturing short- and long-term dependencies in LOB data. The OCET implementation is adapted from Yang, P., Fu, L., Zhang, J., & Li, G. (2022, December). OCET: One-Dimensional Convolution Embedding Transformer for Stock Trend Prediction. In International Conference on Bio-Inspired Computing: Theories and Applications (pp. 370-384). Singapore: Springer Nature Singapore. The OCET implementation is adapted from [Yang et al., 2022].
- **Backtesting Module**: Includes tools to evaluate profitability using metrics such as the Sharpe ratio.

## Architecture
The HFLOB model integrates several key components:
1. **Convolutional Feature Extraction Module**: Captures short-term patterns and local features in LOB data.
2. **OCET Module**: A hybrid architecture combining convolutional layers and self-attention mechanisms for robust feature learning.
3. **Attention Pooling Layer**: Highlights key features for improved interpretability.
4. **Multimodal Fusion Module**: Combines data from different sources into a unified representation.
5. **Final Classification Head**: Outputs predictions for market trends (e.g., upward, stationary, downward).

## Datasets
### FI-2010 Dataset
- Contains LOB data from the Helsinki Stock Exchange for five Finnish stocks.
- Preprocessed with normalization techniques and labeled based on mid-price changes.
- Labels are visualized to ensure consistency and confirm the effectiveness of the labeling approach.

### BTC Perpetual Contract Dataset
- Includes high-frequency data from the Binance exchange, spanning 12 days.
- Features buy/sell order levels and transaction details.

#### Multimodal Data ETL Process
The BTC dataset undergoes the following ETL (Extract, Transform, Load) steps to enable multimodal learning:
1. **Extract**:
   - Raw data files are extracted from the Binance exchange, containing order book levels and transaction details.
   - Data is stored as CSV files for each trading day.

2. **Transform**:
   - **Feature Engineering**: Key features such as order book levels, volume-weighted average price (VWAP), trade flow, cumulative trade flow, and order book imbalance are computed.
   - **Time Features**: Sine and cosine transformations of timestamps are added to capture cyclical patterns.
   - **Market Phases**: Labeled categorical market phases are converted into one-hot encoded features.
   - **Normalization**: StandardScaler is applied to ensure consistent scaling of numerical features.
   - **Data Windowing**: Sequences of 100 time steps are created for model input, aligning features and labels.
   - **EMA-Based Labeling**: BTC price trends are labeled using an exponential moving average approach, providing trend consistency for comparison with FI-2010.

3. **Load**:
   - Processed data is stored in PyTorch tensors, enabling efficient batch loading for training and evaluation.


## Usage
### Data Preparation
- Place BTC perpetual contract datasets in the `data_precess/` directory.
- Run the preprocessing scripts:
  ```bash
  python data_Trade_multi_process.py
  ```

### Training and Testing
- **For BTC Perpetual Contract Dataset**:
  Use the `train_btc_hflob.py` script to train and test the HFLOB model with multimodal features on the BTC dataset:
  ```bash
  python train_btc_hflob.py --use_multimodal
  ```
  To disable multimodal features, omit the `--use_multimodal` flag:
  ```bash
  python train_btc_hflob.py
  ```

### Evaluation
- Perform backtesting:
  ```bash
  python bestmodel_backtesting.py --dataset btc
  ```

## Results
Key findings from the research:
- **Prediction Accuracy**: Achieved 63.40% accuracy on the BTC perpetual contract dataset.
- **Profitability**: Sharpe ratio of 3.48 during backtesting.
- **Labeling Insights**:
  - EMA-based labbackteeling improves prediction accuracy but decreases profitability in backtesting.
  - FI-2010 labeling process is visualized and confirmed for consistency, providing reliable ground truth for training.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions and improvements.

## Citation
If you use this code or find the research helpful, please cite:
```
@mastersthesis{shi2024multimodal,
  author = {Zhong Shi},
  title = {Adaptive Financial Forecasting with Multimodal Learning: Insights from Limit Order Book Data},
  year = {2024},
  institution = {Waikato Institute of Technology},
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
