#数据生成工具
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataGenerator:
    """数据生成工具类"""
    
    def __init__(self):
        # 物理参数配置
        self.physics_params = {
            'temp_supply_range': (5, 10),      # 供水温度范围 (℃)
            'temp_return_range': (10, 15),      # 回水温度范围 (℃)
            'temp_diff_range': (3, 7),          # 供回水温差范围 (℃)
            'flow_range': (80, 120),            # 流量范围 (m³/h)
            'power_range': (150, 250),          # 功率范围 (kW)
            'cop_range': (3.0, 5.0),            # COP范围
            'temp_outdoor_range': (20, 35),     # 室外温度范围 (℃)
            'humidity_range': (40, 80)          # 相对湿度范围 (%)
        }
        
        # 异常模式配置
        self.anomaly_patterns = {
            'sensor_drift': {'probability': 0.05, 'magnitude': 0.5},
            'sudden_change': {'probability': 0.02, 'magnitude': 2.0},
            'noise_increase': {'probability': 0.03, 'std_factor': 3},
            'stuck_value': {'probability': 0.01, 'duration': 12}  # 12个点
        }

    def generate_sample_data(self, output_path='data/sample_data.csv', 
                           periods=1000, freq='5T', add_anomalies=True):
        """
        生成示例数据
        
        参数:
        - output_path: 输出文件路径
        - periods: 数据点数量
        - freq: 采样频率（默认5分钟）
        - add_anomalies: 是否添加异常模式
        """
        try:
            # 生成时间索引
            dates = pd.date_range(start='2025-01-01', periods=periods, freq=freq)
            data = pd.DataFrame(index=dates)
            t = np.arange(len(dates))
            
            # 基础负荷模式（24小时周期）
            daily_pattern = np.sin(2 * np.pi * t / (24 * 12))  # 12点/小时
            weekly_pattern = np.sin(2 * np.pi * t / (24 * 12 * 7))  # 每周变化
            
            # 1. 温度数据
            # 供水温度
            data['temp_supply'] = (
                7 +  # 基准值
                1 * daily_pattern +  # 日变化
                0.5 * weekly_pattern +  # 周变化
                np.random.normal(0, 0.3, len(t))  # 随机波动
            )
            
            # 回水温度 = 供水温度 + 温差
            temp_diff = 5 + 0.5 * daily_pattern + np.random.normal(0, 0.2, len(t))
            data['temp_return'] = data['temp_supply'] + temp_diff
            
            # 2. 流量数据
            data['flow'] = (
                100 +  # 基准流量
                20 * daily_pattern +  # 日变化
                10 * weekly_pattern +  # 周变化
                np.random.normal(0, 3, len(t))  # 随机波动
            )
            
            # 3. 功率数据
            data['power'] = (
                200 +  # 基准功率
                50 * daily_pattern +  # 日变化
                20 * weekly_pattern +  # 周变化
                np.random.normal(0, 5, len(t))  # 随机波动
            )
            
            # 4. 环境数据
            # 室外温度
            data['temp_outdoor'] = (
                25 +  # 基准温度
                5 * daily_pattern +  # 日变化
                3 * weekly_pattern +  # 周变化
                np.random.normal(0, 1, len(t))  # 随机波动
            )
            
            # 相对湿度
            data['humidity_outdoor'] = (
                60 +  # 基准湿度
                10 * daily_pattern +  # 日变化
                5 * weekly_pattern +  # 周变化
                np.random.normal(0, 3, len(t))  # 随机波动
            ).clip(0, 100)  # 限制在0-100范围内
            
            # 5. 计算衍生变量
            data['cooling_load'] = (data['flow'] * 
                              (data['temp_return'] - data['temp_supply']) * 
                              4.2)  # 4.2 为水的比热容
            data['cop'] = data['cooling_load'] / data['power']
            
            # 添加异常模式
            if add_anomalies:
                self._add_anomaly_patterns(data)
            
            # 添加timestamp列
            data['timestamp'] = data.index
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存数据
            data.to_csv(output_path, index=False)
            logger.info(f"示例数据已保存至: {output_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"生成示例数据时出错: {e}")
            return None

    def _add_anomaly_patterns(self, data):
        """添加异常模式到数据中"""
        try:
            # 1. 传感器漂移
            if np.random.random() < self.anomaly_patterns['sensor_drift']['probability']:
                start_idx = np.random.randint(0, len(data) - 100)
                drift = np.linspace(0, self.anomaly_patterns['sensor_drift']['magnitude'], 100)
                data.loc[data.index[start_idx:start_idx+100], 'temp_supply'] += drift
            
            # 2. 突变
            if np.random.random() < self.anomaly_patterns['sudden_change']['probability']:
                idx = np.random.randint(0, len(data))
                data.loc[data.index[idx], 'power'] += (
                    self.anomaly_patterns['sudden_change']['magnitude'] * 
                    data.loc[data.index[idx], 'power']
                )
            
            # 3. 噪声增加
            if np.random.random() < self.anomaly_patterns['noise_increase']['probability']:
                start_idx = np.random.randint(0, len(data) - 50)
                noise = np.random.normal(
                    0, 
                    data['flow'].std() * self.anomaly_patterns['noise_increase']['std_factor'], 
                    50
                )
                data.loc[data.index[start_idx:start_idx+50], 'flow'] += noise
            
            # 4. 传感器卡死
            if np.random.random() < self.anomaly_patterns['stuck_value']['probability']:
                start_idx = np.random.randint(0, len(data) - self.anomaly_patterns['stuck_value']['duration'])
                stuck_value = data.loc[data.index[start_idx], 'temp_return']
                data.loc[data.index[start_idx:start_idx+self.anomaly_patterns['stuck_value']['duration']], 
                        'temp_return'] = stuck_value
            
        except Exception as e:
            logger.error(f"添加异常模式时出错: {e}")

    def generate_batch_data(self, num_files=5, base_path='data/batch'):
        """
        批量生成多个数据文件
        
        参数:
        - num_files: 生成文件数量
        - base_path: 基础保存路径
        """
        try:
            os.makedirs(base_path, exist_ok=True)
            
            for i in range(num_files):
                file_path = os.path.join(base_path, f'sample_data_{i+1}.csv')
                # 为每个文件生成不同的起始时间
                start_date = datetime(2025, 1, 1) + timedelta(days=i*7)
                
                # 生成数据
                self.generate_sample_data(
                    output_path=file_path,
                    periods=1000,
                    freq='5T',
                    add_anomalies=True
                )
                
            logger.info(f"已生成 {num_files} 个示例数据文件")
            
        except Exception as e:
            logger.error(f"批量生成数据时出错: {e}")

    def generate_fault_scenario(self, fault_type, duration_hours=24):
        """
        生成特定故障场景的数据
        
        参数:
        - fault_type: 故障类型
        - duration_hours: 故障持续时间（小时）
        """
        try:
            # 生成基础数据
            periods = duration_hours * 12  # 5分钟间隔
            dates = pd.date_range(start='2025-01-01', periods=periods, freq='5T')
            data = pd.DataFrame(index=dates)
            t = np.arange(len(dates))
            
            # 基础模式
            daily_pattern = np.sin(2 * np.pi * t / (24 * 12))
            
            # 根据故障类型生成数据
            if fault_type == 'compressor_efficiency_decrease':
                # 压缩机效率下降场景
                data['temp_supply'] = 7 + daily_pattern + np.random.normal(0, 0.3, len(t))
                data['temp_return'] = data['temp_supply'] + 5
                data['flow'] = 100 + 20 * daily_pattern
                # 功率逐渐增加
                power_increase = np.linspace(0, 50, len(t))
                data['power'] = 200 + 50 * daily_pattern + power_increase
                
            elif fault_type == 'heat_exchanger_fouling':
                # 换热器结垢场景
                data['temp_supply'] = 7 + daily_pattern + np.random.normal(0, 0.3, len(t))
                # 温差逐渐减小
                temp_diff_decrease = np.linspace(5, 2, len(t))
                data['temp_return'] = data['temp_supply'] + temp_diff_decrease
                data['flow'] = 100 + 20 * daily_pattern
                data['power'] = 200 + 50 * daily_pattern
                
            elif fault_type == 'refrigerant_leak':
                # 制冷剂泄漏场景
                data['temp_supply'] = 7 + np.linspace(0, 3, len(t)) + daily_pattern
                data['temp_return'] = data['temp_supply'] + 5
                data['flow'] = 100 + 20 * daily_pattern
                # COP逐渐降低
                data['power'] = 200 + np.linspace(0, 100, len(t)) + 50 * daily_pattern
            
            # 添加环境数据
            data['temp_outdoor'] = 25 + 5 * daily_pattern + np.random.normal(0, 1, len(t))
            data['humidity_outdoor'] = 60 + 10 * daily_pattern + np.random.normal(0, 3, len(t))
            
            # 添加故障标签
            data['fault_type'] = fault_type
            
            return data
            
        except Exception as e:
            logger.error(f"生成故障场景数据时出错: {e}")
            return None

if __name__ == "__main__":
    # 使用示例
    generator = DataGenerator()
    
    # 生成单个示例数据文件
    data = generator.generate_sample_data()
    
    # 批量生成数据文件
    generator.generate_batch_data(num_files=3)
    
    # 生成故障场景数据
    fault_data = generator.generate_fault_scenario('compressor_efficiency_decrease')
    if fault_data is not None:
        fault_data.to_csv('data/fault_scenario.csv', index=False)