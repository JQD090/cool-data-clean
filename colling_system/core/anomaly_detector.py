import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import shap
import logging
from .physics_model import PhysicsConstraintModel

logger = logging.getLogger(__name__)

class LSTMAnomalyDetector(nn.Module):
    """LSTM自编码器用于时序异常检测"""
    
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # 编码
        out, (hidden, cell) = self.encoder(x)
        
        # 解码
        out, _ = self.decoder(out, (hidden, cell))
        
        # 重构
        out = self.fc(out)
        return out

class MultiModalAnomalyDetector:
    """多模态异常检测器，集成多种检测方法"""
    
    def __init__(self, config=None):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.physics_model = PhysicsConstraintModel()
        
        # 异常检测参数
        self.anomaly_params = {
            'lstm_sequence_length': 30,     # LSTM序列长度
            'lstm_hidden_dim': 32,          # LSTM隐藏层维度
            'isolation_forest_contamination': 0.1,  # 孤立森林异常比例
            'residual_threshold': 2.0,      # 残差异常阈值(标准差倍数)
            'min_train_samples': 1000,      # 最小训练样本数
        }

    def train_lstm_model(self, data, feature_cols, seq_len=None, 
                        epochs=10, batch_size=64, lr=1e-3):
        """
        训练LSTM异常检测模型
        
        参数:
        - data: DataFrame格式的训练数据
        - feature_cols: 特征列名列表
        - seq_len: 序列长度，默认使用配置值
        - epochs: 训练轮数
        - batch_size: 批次大小
        - lr: 学习率
        """
        try:
            if seq_len is None:
                seq_len = self.anomaly_params['lstm_sequence_length']
            
            # 数据准备
            X = data[feature_cols].values
            self.lstm_scaler = StandardScaler()
            X_scaled = self.lstm_scaler.fit_transform(X)
            
            # 创建序列数据
            sequences = []
            for i in range(len(X_scaled) - seq_len):
                sequences.append(X_scaled[i:i+seq_len])
            sequences = np.array(sequences)
            
            # 转换为PyTorch张量
            X_tensor = torch.FloatTensor(sequences)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 初始化模型
            self.lstm_model = LSTMAnomalyDetector(
                input_dim=len(feature_cols),
                hidden_dim=self.anomaly_params['lstm_hidden_dim']
            )
            
            # 训练
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            self.lstm_model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    output = self.lstm_model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], '
                              f'Loss: {total_loss/len(dataloader):.4f}')
            
            self.lstm_feature_cols = feature_cols
            self.lstm_seq_len = seq_len
            logger.info("LSTM模型训练完成")
            return True
            
        except Exception as e:
            logger.error(f"LSTM模型训练失败: {e}")
            return False

    def detect_lstm_anomalies(self, data, threshold=None, method='std'):
        """
        使用LSTM模型检测异常
        
        参数:
        - data: 待检测数据
        - threshold: 异常阈值
        - method: 'std'（均值+2*std）或'quantile'（95分位）
        
        返回:
        - 异常点索引列表
        """
        if not hasattr(self, 'lstm_model'):
            logger.error("LSTM模型未训练")
            return []
        
        try:
            # 准备数据
            X = data[self.lstm_feature_cols].values
            X_scaled = self.lstm_scaler.transform(X)
            
            # 创建序列
            sequences = []
            for i in range(len(X_scaled) - self.lstm_seq_len):
                sequences.append(X_scaled[i:i+self.lstm_seq_len])
            sequences = np.array(sequences)
            
            # 预测
            self.lstm_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(sequences)
                reconstructed = self.lstm_model(X_tensor).numpy()
            
            # 计算重构误差
            errors = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))

            # 确定阈值
            if threshold is None:
                if method == 'quantile':
                    threshold = np.quantile(errors, 0.95)
                else:
                    threshold = np.mean(errors) + 2 * np.std(errors)
            
            # 检测异常
            anomaly_indices = np.where(errors > threshold)[0]
            
            # 映射回原始数据索引
            anomaly_indices = [i + self.lstm_seq_len for i in anomaly_indices]
            
            return data.index[anomaly_indices].tolist()
            
        except Exception as e:
            logger.error(f"LSTM异常检测失败: {e}")
            return []

    def detect_statistical_anomalies(self, data):
        """使用统计方法检测异常"""
        try:
            # 选择数值列并使用新的填充方法
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_cols].ffill().bfill()  # 修改这里
            
            # 使用孤立森林检测异常
            iso_forest = IsolationForest(
                contamination=self.anomaly_params['isolation_forest_contamination'],
                random_state=42
            )
            
            labels = iso_forest.fit_predict(X)
            anomaly_indices = data.index[labels == -1].tolist()
            
            return anomaly_indices
        
        except Exception as e:
            logger.error(f"统计异常检测失败: {e}")
            return []

    def detect_comprehensive_anomalies(self, data):
        """
        综合异常检测：结合多种方法
        
        返回:
        - dict: 包含各种方法检测到的异常
        """
        results = {
            'lstm_anomalies': [],
            'statistical_anomalies': [],
            'physical_anomalies': {},
            'combined_anomalies': set()
        }
        
        try:
            # 1. LSTM异常检测
            if hasattr(self, 'lstm_model'):
                lstm_anomalies = self.detect_lstm_anomalies(data)
                results['lstm_anomalies'] = lstm_anomalies
                results['combined_anomalies'].update(lstm_anomalies)
            
            # 2. 统计异常检测
            statistical_anomalies = self.detect_statistical_anomalies(data)
            results['statistical_anomalies'] = statistical_anomalies
            results['combined_anomalies'].update(statistical_anomalies)
            
            # 3. 物理异常检测
            physical_anomalies = self.physics_model.detect_physical_anomalies(data)
            results['physical_anomalies'] = physical_anomalies
            for anomaly_list in physical_anomalies.values():
                results['combined_anomalies'].update(anomaly_list)
            
            # 转换集合为列表
            results['combined_anomalies'] = sorted(list(results['combined_anomalies']))
            
            return results
            
        except Exception as e:
            logger.error(f"综合异常检测失败: {e}")
            return results

    def explain_anomalies(self, data, anomaly_indices):
        """
        解释检测到的异常
        
        参数:
        - data: 原始数据
        - anomaly_indices: 异常点索引列表
        
        返回:
        - dict: 异常解释结果
        """
        explanations = {}
        
        try:
            for idx in anomaly_indices:
                if not isinstance(idx, pd.Timestamp):
                    continue
                
                explanation = {
                    'timestamp': idx,
                    'values': {},
                    'contributions': {},
                    'type': set()
                }
                
                # 1. 记录异常点的值
                row = data.loc[idx]
                for col in data.columns:
                    explanation['values'][col] = row[col]
                
                # 2. LSTM解释（如果可用）
                if hasattr(self, 'lstm_model'):
                    lstm_contrib = self._explain_lstm_anomaly(data, idx)
                    if lstm_contrib:
                        explanation['contributions']['lstm'] = lstm_contrib
                        
                # 3. 物理规则解释
                physics_result = self.physics_model.online_anomaly_detection(row)
                if physics_result['physical_flags']:
                    explanation['contributions']['physics'] = physics_result
                    explanation['type'].update(physics_result['physical_flags'].keys())
                
                # 4. 统计特征解释
                stats_contrib = self._explain_statistical_anomaly(data, idx)
                if stats_contrib:
                    explanation['contributions']['statistical'] = stats_contrib
                
                explanations[idx] = explanation
            
            return explanations
            
        except Exception as e:
            logger.error(f"异常解释失败: {e}")
            return explanations

    def _explain_lstm_anomaly(self, data, idx):
        """使用SHAP解释LSTM异常"""
        if not hasattr(self, 'lstm_model'):
            return None
        
        try:
            # 准备解释器的背景数据
            background_data = data[self.lstm_feature_cols].values
            background_data = self.lstm_scaler.transform(background_data)
            
            # 创建解释器
            explainer = shap.DeepExplainer(self.lstm_model, 
                                         torch.FloatTensor(background_data))
            
            # 获取SHAP值
            target_data = data.loc[idx:idx, self.lstm_feature_cols]
            target_scaled = self.lstm_scaler.transform(target_data)
            shap_values = explainer.shap_values(torch.FloatTensor(target_scaled))
            
            # 处理SHAP值
            feature_importance = {}
            for i, feature in enumerate(self.lstm_feature_cols):
                feature_importance[feature] = abs(shap_values[0][i]).mean()
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"LSTM异常解释失败: {e}")
            return None

    def _explain_statistical_anomaly(self, data, idx):
        """统计特征解释"""
        try:
            # 计算z-scores
            row = data.loc[idx]
            z_scores = {}
            
            for col in data.select_dtypes(include=[np.number]).columns:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    z_scores[col] = abs((row[col] - mean) / std)
            
            return {
                'z_scores': z_scores,
                'extreme_features': [col for col, z in z_scores.items() if z > 3]
            }
            
        except Exception as e:
            logger.error(f"统计异常解释失败: {e}")
            return None

    def save_lstm_model(self, path="lstm_model.pth"):
        """保存LSTM模型和Scaler"""
        if hasattr(self, 'lstm_model'):
            torch.save({
                'model_state_dict': self.lstm_model.state_dict(),
                'scaler': self.lstm_scaler,
                'feature_cols': self.lstm_feature_cols,
                'seq_len': self.lstm_seq_len
            }, path)
            logger.info(f"LSTM模型已保存至: {path}")

    def load_lstm_model(self, path="lstm_model.pth"):
        """加载LSTM模型和Scaler"""
        checkpoint = torch.load(path)
        self.lstm_model = LSTMAnomalyDetector(
            input_dim=len(checkpoint['feature_cols']),
            hidden_dim=self.anomaly_params['lstm_hidden_dim']
        )
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_scaler = checkpoint['scaler']
        self.lstm_feature_cols = checkpoint['feature_cols']
        self.lstm_seq_len = checkpoint['seq_len']
        logger.info(f"LSTM模型已从{path}加载")

if __name__ == "__main__":
    # 使用示例
    detector = MultiModalAnomalyDetector()
    
    # 创建示例数据
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5T')
    data = pd.DataFrame(index=dates)
    data['temp'] = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    data['power'] = np.cos(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    
    # 训练模型
    detector.train_lstm_model(data, ['temp', 'power'])
    
    # 检测异常
    anomalies = detector.detect_comprehensive_anomalies(data)
    print(f"检测到 {len(anomalies['combined_anomalies'])} 个异常点")