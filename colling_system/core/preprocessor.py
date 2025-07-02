#数据预处理模块
import numpy as np
import pandas as pd
from scipy import stats
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理与质量分析模块"""
    
    def __init__(self, config=None):
        self.config = config
        
        # 设置绘图样式
        plt.style.use('default')  # 使用默认样式
        sns.set_style("whitegrid")  # 使用seaborn的whitegrid样式
        
        self.sensor_types = {
            'temperature': {'unit': '℃', 'type': 'analog', 'frequency': '1-5min'},
            'power': {'unit': 'kW', 'type': 'digital', 'frequency': '1min'},
            'flow': {'unit': 'm³/h', 'type': 'digital', 'frequency': '1min'},
            'pressure': {'unit': 'MPa', 'type': 'analog', 'frequency': '5min'},
            'valve_opening': {'unit': '%', 'type': 'control', 'frequency': 'variable'},
            'environment': {'unit': '℃/RH%', 'type': 'external', 'frequency': '15min-1h'}
        }
        
    def load_data(self, file_path):
        """加载传感器数据"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path, parse_dates=['timestamp'])
            else:
                data = pd.read_excel(file_path, parse_dates=['timestamp'])
            if 'timestamp' not in data.columns:
                raise ValueError("数据文件缺少'timestamp'字段")
            data.set_index('timestamp', inplace=True)
            return data
        except Exception as e:
            logger.error(f"数据加载错误: {e}")
            return None
    
    def check_data_quality(self, data):
        """数据质量检查"""
        quality_report = {}
        
        for col in data.columns:
            report = {
                'missing_rate': data[col].isna().sum() / len(data),
                'zero_rate': (data[col] == 0).sum() / len(data),
                'constant_rate': self._check_constant_values(data[col]),
                'outlier_rate': self._detect_outliers(data[col]),
                'noise_level': self._estimate_noise_level(data[col])
            }
            quality_report[col] = report
            
        return quality_report
    
    def _check_constant_values(self, series):
        """检查恒定值比例"""
        if len(series) < 2:
            return 0
        diff = series.diff().dropna()
        return (diff == 0).sum() / len(diff)
    
    def _detect_outliers(self, series):
        """检测异常值比例"""
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return 0
        q1, q3 = clean_series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((clean_series < q1 - 1.5 * iqr) | (clean_series > q3 + 1.5 * iqr)).sum()
        return outliers / len(clean_series)
    
    def _estimate_noise_level(self, series):
        """估计噪声水平"""
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return 0
        # 使用移动平均估计信号，残差作为噪声
        smooth = clean_series.rolling(window=5, center=True).mean()
        noise = (clean_series - smooth).dropna()
        return noise.std() / clean_series.std() if clean_series.std() > 0 else 0
    
    def clean_data(self, data, quality_report):
        """基础数据清洗"""
        cleaned_data = data.copy()
        
        for col in cleaned_data.columns:
            # 插值处理缺失值
            if quality_report[col]['missing_rate'] < 0.3:
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear')
            
            # 移除明显异常值
            if quality_report[col]['outlier_rate'] > 0.01:
                q1, q3 = cleaned_data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                mask = (cleaned_data[col] >= q1 - 3 * iqr) & (cleaned_data[col] <= q3 + 3 * iqr)
                cleaned_data.loc[~mask, col] = np.nan
                cleaned_data[col] = cleaned_data[col].interpolate(method='linear')
        
        return cleaned_data



    def detect_sampling_frequency(self, data):
        """
        根据时间戳自动检测数据采样频率
        参数:
        - data: 带时间戳索引的DataFrame
        返回:
        - dict: 每个传感器的采样频率（秒）
        """
        sampling_freq = {}
        
        for col in data.columns:
            # 获取非空数据的时间戳
            valid_timestamps = data[col].dropna().index
            
            if len(valid_timestamps) < 2:
                sampling_freq[col] = None
                continue
                
            # 计算相邻时间戳的差值
            time_diffs = np.diff(valid_timestamps) / np.timedelta64(1, 's')
            
            # 使用众数作为采样频率（秒）
            mode_freq = stats.mode(time_diffs, keepdims=True)[0][0]
            
            # 转换为人类可读的格式
            if mode_freq < 60:
                freq_str = f"{mode_freq:.0f}秒"
            elif mode_freq < 3600:
                freq_str = f"{mode_freq/60:.1f}分钟"
            else:
                freq_str = f"{mode_freq/3600:.1f}小时"
                
            sampling_freq[col] = {
                'seconds': mode_freq,
                'readable': freq_str
            }
        
        return sampling_freq