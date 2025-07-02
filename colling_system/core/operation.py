#相似工况模块
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import logging

logger = logging.getLogger(__name__)

class SimilarOperationIdentifier:
    """相似工况识别模块"""
    
    def __init__(self, config=None):
        """初始化相似工况识别器"""
        # 定义可选的特征列
        self.all_feature_cols = {
            'primary': [  # 必需特征
                'temp_supply',
                'temp_return',
                'flow',
                'power'
            ],
            'secondary': [  # 可选特征
                'cooling_load',
                'temp_outdoor',
                'humidity_outdoor',
                'cop'
            ]
        }
        
        # 特征权重配置
        self.feature_weights = {
            'temp_supply': 0.8,
            'temp_return': 0.8,
            'flow': 0.7,
            'power': 0.9,
            'cooling_load': 1.0,
            'temp_outdoor': 0.6,
            'humidity_outdoor': 0.4,
            'cop': 0.9
        }
        
        self.similarity_threshold = 0.85
        self.typical_operations = {}
        self.scaler = None

    def build_operation_features(self, data, window_size=12):
        """
        构建工况特征向量
        
        参数:
        - data: DataFrame 原始数据
        - window_size: int 滑动窗口大小
        
        返回:
        - DataFrame 特征矩阵
        """
        try:
            # 确定可用的特征列
            available_cols = [col for col in self.all_feature_cols['primary'] 
                            if col in data.columns]
            available_cols.extend([col for col in self.all_feature_cols['secondary'] 
                                 if col in data.columns])
            
            if not available_cols:
                raise ValueError("没有找到任何可用的特征列")
            
            # 计算派生特征
            features = data[available_cols].copy()
            
            # 计算冷量（如果缺少但有必要的数据）
            if 'cooling_load' not in features.columns and all(col in data.columns 
                for col in ['flow', 'temp_supply', 'temp_return']):
                features['cooling_load'] = (data['flow'] * 
                                         (data['temp_return'] - data['temp_supply']) * 
                                         4.2)
            
            # 计算COP（如果缺少但有必要的数据）
            if 'cop' not in features.columns and all(col in data.columns 
                for col in ['cooling_load', 'power']):
                features['cop'] = data['cooling_load'] / (data['power'] + 1e-6)
            
            # 添加动态特征
            for col in features.columns:
                # 计算滑动窗口内的趋势（斜率）
                slope = features[col].rolling(window=window_size).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                    if len(x) >= window_size/2 else np.nan
                )
                features[f'{col}_trend'] = slope
                
                # 计算滑动窗口内的波动性
                std = features[col].rolling(window=window_size).std()
                features[f'{col}_volatility'] = std / features[col].abs().mean()
            
            # 标准化特征
            if self.scaler is None:
                self.scaler = StandardScaler()
                features_scaled = pd.DataFrame(
                    self.scaler.fit_transform(features.fillna(0)),
                    columns=features.columns,
                    index=features.index
                )
            else:
                features_scaled = pd.DataFrame(
                    self.scaler.transform(features.fillna(0)),
                    columns=features.columns,
                    index=features.index
                )
            
            return features_scaled
            
        except Exception as e:
            logger.error(f"构建工况特征时出错: {e}")
            return None

    def calculate_similarity(self, current_features, historical_features):
        """
        计算相似度
        
        参数:
        - current_features: Series 当前工况特征
        - historical_features: Series 历史工况特征
        
        返回:
        - float 相似度得分
        """
        try:
            # 获取共同的特征列
            common_cols = [col for col in current_features.index 
                          if col in historical_features.index]
            
            if not common_cols:
                return 0.0
            
            # 获取权重
            weights = np.array([self.feature_weights.get(col.split('_')[0], 0.5) 
                              for col in common_cols])
            
            # 提取共同特征
            current = current_features[common_cols]
            historical = historical_features[common_cols]
            
            # 应用权重
            current_weighted = current * weights
            historical_weighted = historical * weights
            
            # 计算余弦相似度
            similarity = np.dot(current_weighted, historical_weighted) / (
                np.linalg.norm(current_weighted) * np.linalg.norm(historical_weighted)
            )
            
            return max(0, min(1, similarity))  # 确保在[0,1]范围内
            
        except Exception as e:
            logger.error(f"计算相似度时出错: {e}")
            return 0.0

    def identify_similar_operations(self, current_data, historical_data, 
                                  n_similar=5, time_window='7D'):
        """
        识别与当前工况相似的历史工况
        
        参数:
        - current_data: DataFrame 当前工况数据
        - historical_data: DataFrame 历史数据
        - n_similar: int 返回最相似的个数
        - time_window: str 时间窗口大小
        
        返回:
        - list 相似工况列表
        """
        try:
            # 准备当前工况特征
            current_features = self.build_operation_features(current_data)
            if current_features is None:
                return []
            current_vector = current_features.iloc[-1]
            
            # 在历史数据中搜索相似工况
            historical_features = self.build_operation_features(historical_data)
            if historical_features is None:
                return []
            
            # 计算相似度
            similarities = []
            for idx, hist_vector in historical_features.iterrows():
                # 检查时间窗口
                if pd.Timestamp(idx) > pd.Timestamp(current_data.index[-1]) - pd.Timedelta(time_window):
                    continue
                
                sim = self.calculate_similarity(current_vector, hist_vector)
                if sim >= self.similarity_threshold:
                    similarities.append({
                        'timestamp': idx,
                        'similarity': sim,
                        'features': hist_vector
                    })
            
            # 排序并返回最相似的n个工况
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:n_similar]
            
        except Exception as e:
            logger.error(f"识别相似工况时出错: {e}")
            return []

    def cluster_typical_operations(self, historical_data, n_clusters=5):
        """
        对历史工况进行聚类，识别典型工况模式
        
        参数:
        - historical_data: DataFrame 历史数据
        - n_clusters: int 聚类数量
        
        返回:
        - dict 典型工况信息
        """
        try:
            # 构建特征矩阵
            features = self.build_operation_features(historical_data)
            if features is None:
                return {}
            
            # 执行聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(features)
            
            # 保存典型工况
            self.typical_operations = {}
            for i in range(n_clusters):
                cluster_data = historical_data[labels == i]
                
                # 计算聚类中心的实际工况参数
                center_actual = {}
                for col in historical_data.columns:
                    center_actual[col] = cluster_data[col].mean()
                
                self.typical_operations[f'cluster_{i}'] = {
                    'center': kmeans.cluster_centers_[i],
                    'center_actual': center_actual,
                    'samples': cluster_data.index.tolist(),
                    'size': len(cluster_data),
                    'variance': cluster_data.var().to_dict(),
                    'time_distribution': self._analyze_time_distribution(cluster_data)
                }
            
            return self.typical_operations
            
        except Exception as e:
            logger.error(f"聚类典型工况时出错: {e}")
            return {}

    def _analyze_time_distribution(self, cluster_data):
        """分析工况的时间分布特征"""
        return {
            'hour_dist': cluster_data.index.hour.value_counts().to_dict(),
            'weekday_dist': cluster_data.index.dayofweek.value_counts().to_dict()
        }

    def find_optimal_operations(self, historical_data):
        """
        在历史数据中查找最优运行工况
        
        参数:
        - historical_data: DataFrame 历史数据
        
        返回:
        - dict 最优工况信息
        """
        try:
            if 'cop' not in historical_data.columns:
                if all(col in historical_data.columns 
                      for col in ['cooling_load', 'power']):
                    historical_data['cop'] = (historical_data['cooling_load'] / 
                                           (historical_data['power'] + 1e-6))
                else:
                    logger.warning("缺少必要的数据计算COP")
                    return {}
            
            # 按不同负荷区间查找最优工况
            load_ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]  # 低、中、高负荷
            optimal_operations = {}
            
            max_load = historical_data['cooling_load'].max()
            
            for load_range in load_ranges:
                load_min = load_range[0] * max_load
                load_max = load_range[1] * max_load
                
                # 筛选负荷区间的数据
                range_data = historical_data[
                    (historical_data['cooling_load'] >= load_min) & 
                    (historical_data['cooling_load'] < load_max)
                ]
                
                if len(range_data) == 0:
                    continue
                
                # 找出该负荷区间COP最高的工况
                best_cop_idx = range_data['cop'].idxmax()
                optimal_operations[f'load_range_{load_range[0]}_{load_range[1]}'] = {
                    'timestamp': best_cop_idx,
                    'cop': range_data.loc[best_cop_idx, 'cop'],
                    'load': range_data.loc[best_cop_idx, 'cooling_load'],
                    'parameters': range_data.loc[best_cop_idx].to_dict()
                }
            
            return optimal_operations
            
        except Exception as e:
            logger.error(f"查找最优工况时出错: {e}")
            return {}

    def analyze_operation_patterns(self, historical_data):
        """
        分析运行模式
        
        参数:
        - historical_data: DataFrame 历史数据
        
        返回:
        - dict 运行模式分析结果
        """
        try:
            patterns = {
                'daily_patterns': {},
                'weekly_patterns': {},
                'load_patterns': {}
            }
            
            # 1. 日内模式分析
            for hour in range(24):
                hour_data = historical_data[historical_data.index.hour == hour]
                if len(hour_data) > 0:
                    patterns['daily_patterns'][hour] = {
                        'avg_load': hour_data['cooling_load'].mean(),
                        'avg_cop': hour_data['cop'].mean() if 'cop' in hour_data else None,
                        'samples': len(hour_data)
                    }
            
            # 2. 周内模式分析
            for day in range(7):
                day_data = historical_data[historical_data.index.dayofweek == day]
                if len(day_data) > 0:
                    patterns['weekly_patterns'][day] = {
                        'avg_load': day_data['cooling_load'].mean(),
                        'avg_cop': day_data['cop'].mean() if 'cop' in day_data else None,
                        'samples': len(day_data)
                    }
            
            # 3. 负荷特征分析
            if 'cooling_load' in historical_data:
                load_max = historical_data['cooling_load'].max()
                load_ranges = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]
                
                for load_min, load_max in load_ranges:
                    range_data = historical_data[
                        (historical_data['cooling_load'] >= load_min * load_max) & 
                        (historical_data['cooling_load'] < load_max * load_max)
                    ]
                    if len(range_data) > 0:
                        patterns['load_patterns'][f'{load_min}-{load_max}'] = {
                            'samples': len(range_data),
                            'avg_cop': range_data['cop'].mean() if 'cop' in range_data else None,
                            'typical_conditions': self._get_typical_conditions(range_data)
                        }
            
            return patterns
            
        except Exception as e:
            logger.error(f"分析运行模式时出错: {e}")
            return {}

    def _get_typical_conditions(self, data):
        """获取典型运行条件"""
        return {
            col: {
                'mean': data[col].mean(),
                'std': data[col].std()
            }
            for col in self.all_feature_cols['primary'] 
            if col in data.columns
        }

    def analyze_similar_operations(self, data):
        """
        分析相似工况
    
        参数:
            data: DataFrame, 包含运行数据
        
        返回:
            dict: 工况分析结果
        """
        try:
            results = {
                'typical_operations': {},
                'optimal_operations': {},
                'operation_patterns': {}
            }
            
            # 1. 识别典型工况
            typical_ops = self.cluster_typical_operations(data)
            results['typical_operations'] = typical_ops
            
            # 2. 查找最优工况
            optimal_ops = self.find_optimal_operations(data)
            results['optimal_operations'] = optimal_ops
            
            # 3. 分析运行模式
            patterns = self.analyze_operation_patterns(data)
            results['operation_patterns'] = patterns
            
            # 4. 添加统计信息
            results['statistics'] = {
                'total_patterns': len(typical_ops),
                'optimal_count': len(optimal_ops),
                'analysis_timestamp': pd.Timestamp.now()
            }
            
            return results
        
        except Exception as e:
            logger.error(f"分析相似工况时出错: {e}")
            return {
                'typical_operations': {},
                'optimal_operations': {},
                'operation_patterns': {},
                'error': str(e)
            }

if __name__ == "__main__":
    # 使用示例
    identifier = SimilarOperationIdentifier()
    
    # 生成示例数据
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='5T')
    data = pd.DataFrame(index=dates)
    
    # 模拟一些运行数据
    data['temp_supply'] = 7 + np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.3, 1000)
    data['temp_return'] = data['temp_supply'] + 5 + np.random.normal(0, 0.2, 1000)
    data['flow'] = 100 + 20*np.sin(np.linspace(0, 8*np.pi, 1000)) + np.random.normal(0, 5, 1000)
    data['power'] = 200 + 50*np.sin(np.linspace(0, 6*np.pi, 1000)) + np.random.normal(0, 10, 1000)
    
    # 识别典型工况
    typical_ops = identifier.cluster_typical_operations(data)
    print("\n典型工况聚类结果:")
    for cluster_id, cluster_info in typical_ops.items():
        print(f"\n{cluster_id}:")
        print(f"样本数量: {cluster_info['size']}")
        print(f"中心点特征: {cluster_info['center_actual']}")