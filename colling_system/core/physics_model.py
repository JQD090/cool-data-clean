#物理约束模块
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

class PhysicsConstraintModel:
    """物理约束建模模块"""
    
    def __init__(self, config=None):  
        """初始化物理约束模型
        
        参数:
            config: 配置对象，可选
        """
        self.config = config
        
        # 物理常数
        self.water_density = 1000  # kg/m³
        self.water_specific_heat = 4.2  # kJ/(kg·℃)
        self.system_efficiency_range = (0.6, 0.8)  # 系统效率范围
        
        # 正常运行范围
        self.normal_ranges = {
            'temp_supply': (5, 10),      # 供水温度正常范围 ℃
            'temp_return': (10, 15),     # 回水温度正常范围 ℃
            'temp_diff': (3, 7),         # 供回水温差正常范围 ℃
            'cop_range': (3.0, 5.0),     # COP正常范围
            'cooling_water_temp_in': (30, 35),   # 冷却水进水温度 ℃
            'cooling_water_temp_out': (35, 40),  # 冷却水出水温度 ℃
        }
        
        # 异常判定阈值
        self.anomaly_thresholds = {
            'temp_residual': 2.0,      # 温度残差阈值 ℃
            'power_residual': 30.0,    # 功率残差阈值 kW
            'cooling_residual': 5.0,   # 冷量残差阈值 %
            'temp_diff_min': 0.1,      # 供回水温差最小值
            'cop_abnormal_low': 2.0,   # COP异常低值
            'cop_abnormal_high': 6.0,  # COP异常高值
        }

    def calculate_cooling_capacity(self, flow, temp_return, temp_supply):
        """
        计算冷量 Q = ρ × c × F × (Tr - Ts)
        
        参数:
        - flow: 冷冻水流量 (m³/s)
        - temp_return: 冷冻回水温度 (℃)
        - temp_supply: 冷冻供水温度 (℃)
        返回:
        - float: 制冷量 (kW)
        """
        try:
            # 参数验证
            if pd.isna(flow) or pd.isna(temp_return) or pd.isna(temp_supply):
                return np.nan
            
            if flow <= 0:
                return np.nan
            
            temp_diff = temp_return - temp_supply
            if temp_diff <= 0:
                return np.nan
            
            # 冷量计算 (kW)
            q_evap = self.water_density * self.water_specific_heat * flow * temp_diff
            
            return q_evap
        except Exception as e:
            logger.error(f"计算冷量时出错: {e}")
            return np.nan

    def calculate_condenser_heat_rejection(self, flow_cooling, temp_out, temp_in):
        """
        计算冷凝器散热量 Qc = ρ × c × Fc × (Tout - Tin)
        
        参数:
        - flow_cooling: 冷却水流量 (m³/s)
        - temp_out: 冷却水出水温度 (℃)
        - temp_in: 冷却水入水温度 (℃)
        返回:
        - float: 散热量 (kW)
        """
        try:
            if pd.isna(flow_cooling) or pd.isna(temp_out) or pd.isna(temp_in):
                return np.nan
            
            if flow_cooling <= 0:
                return np.nan
            
            temp_diff = temp_out - temp_in
            if temp_diff <= 0:
                return np.nan
            
            q_reject = self.water_density * self.water_specific_heat * flow_cooling * temp_diff
            
            return q_reject
        except Exception as e:
            logger.error(f"计算冷凝器散热量时出错: {e}")
            return np.nan

    def calculate_cop(self, q_evap, power):
        """
        计算COP = Q实际 / P实际
        """
        try:
            if pd.isna(q_evap) or pd.isna(power) or power <= 0:
                return np.nan
            return q_evap / power
        except Exception as e:
            logger.error(f"计算COP时出错: {e}")
            return np.nan

    def validate_energy_balance(self, power_input, q_evap, q_reject):
        """
        验证能量守恒 P压缩机 = Qc - Qe
        """
        try:
            if pd.isna(power_input) or pd.isna(q_evap) or pd.isna(q_reject):
                return False, np.nan
            
            # 理论压缩机功率
            w_theoretical = q_reject - q_evap
            
            # 计算系统效率
            if w_theoretical > 0:
                efficiency = w_theoretical / power_input
            else:
                return False, 0
            
            # 检查效率是否在合理范围内
            is_valid = (self.system_efficiency_range[0] <= efficiency <= 
                       self.system_efficiency_range[1])
            
            return is_valid, efficiency
        except Exception as e:
            logger.error(f"验证能量平衡时出错: {e}")
            return False, np.nan

    def calculate_residuals(self, data):
        """计算各种残差指标"""
        try:
            residuals = {}
            
            # 计算实际冷量
            if all(col in data.columns for col in ['flow', 'temp_return', 'temp_supply']):
                q_actual = []
                for _, row in data.iterrows():
                    q = self.calculate_cooling_capacity(
                        row['flow'] / 3600,  # 转换为 m³/s
                        row['temp_return'],
                        row['temp_supply']
                    )
                    q_actual.append(q)
                residuals['q_actual'] = pd.Series(q_actual, index=data.index)
            
            # 计算COP实际值
            if 'power' in data.columns and 'q_actual' in residuals:
                cop_actual = residuals['q_actual'] / data['power']
                residuals['cop_actual'] = cop_actual
            
            # 计算功率残差
            if 'power' in data.columns and 'q_actual' in residuals:
                standard_cop = 3.5  # 标准COP值
                p_estimated = residuals['q_actual'] / standard_cop
                residuals['power_residual'] = data['power'] - p_estimated
            
            # 计算温度残差
            for temp_col in ['temp_supply', 'temp_return']:
                if temp_col in data.columns:
                    temp_mean = data[temp_col].rolling(window=20, center=True).mean()
                    residuals[f'{temp_col}_residual'] = data[temp_col] - temp_mean
            
            return residuals
        except Exception as e:
            logger.error(f"计算残差时出错: {e}")
            return {}

    def detect_physical_anomalies(self, data):
        """基于物理异常判定矩阵检测异常"""
        try:
            anomalies = {
                'sensor_drift': [],          # 传感器漂移
                'compressor_efficiency': [], # 压缩机效率下降
                'condenser_fouling': [],     # 冷凝器换热不良
                'evaporator_fouling': [],    # 蒸发器换热异常
                'refrigerant_leak': [],      # 制冷剂不足/泄漏
                'sensor_stuck': [],          # 传感器卡死
                'data_coupling_conflict': [],# 数据耦合冲突
                'modeling_error': []         # 稳态建模误差
            }
            
            # 计算残差
            residuals = self.calculate_residuals(data)
            
            for idx, row in data.iterrows():
                anomaly_flags = []
                
                # 获取当前行的残差值
                temp_residual = abs(residuals['temp_supply_residual'].loc[idx]) if 'temp_supply_residual' in residuals and not pd.isna(residuals['temp_supply_residual'].loc[idx]) else 0
                power_residual = abs(residuals['power_residual'].loc[idx]) if 'power_residual' in residuals and not pd.isna(residuals['power_residual'].loc[idx]) else 0
                cop_value = residuals['cop_actual'].loc[idx] if 'cop_actual' in residuals else np.nan
                temp_diff = row['temp_return'] - row['temp_supply'] if all(col in row.index for col in ['temp_return', 'temp_supply']) else 0
                
                # 应用判定规则
                self._apply_anomaly_rules(
                    anomalies, idx, temp_residual, power_residual,
                    cop_value, temp_diff
                )
            
            return anomalies
        except Exception as e:
            logger.error(f"检测物理异常时出错: {e}")
            return {}

    def _apply_anomaly_rules(self, anomalies, idx, temp_residual, power_residual, 
                            cop_value, temp_diff):
        """应用异常判定规则"""
        try:
            # 1. 传感器漂移
            if (temp_residual > self.anomaly_thresholds['temp_residual'] and 
                power_residual < 10 and 
                not pd.isna(cop_value) and 
                self.normal_ranges['cop_range'][0] <= cop_value <= self.normal_ranges['cop_range'][1]):
                anomalies['sensor_drift'].append(idx)
            
            # 2. 压缩机效率下降
            if (temp_residual > self.anomaly_thresholds['temp_residual'] and 
                power_residual > 50 and 
                not pd.isna(cop_value) and 
                cop_value < self.normal_ranges['cop_range'][0]):
                anomalies['compressor_efficiency'].append(idx)
            
            # 3. 冷凝器换热不良
            if (power_residual > self.anomaly_thresholds['power_residual'] and 
                not pd.isna(cop_value) and 
                cop_value < self.normal_ranges['cop_range'][0]):
                anomalies['condenser_fouling'].append(idx)
            
            # 4. 蒸发器换热异常
            if temp_diff < self.normal_ranges['temp_diff'][0]:
                anomalies['evaporator_fouling'].append(idx)
            
            # 5. 制冷剂不足/泄漏
            if (power_residual > 20 and 
                not pd.isna(cop_value) and 
                cop_value < self.anomaly_thresholds['cop_abnormal_low']):
                anomalies['refrigerant_leak'].append(idx)
            
            # 6. 传感器卡死
            if temp_diff < self.anomaly_thresholds['temp_diff_min']:
                anomalies['sensor_stuck'].append(idx)
            
            # 7. COP异常判断
            if (not pd.isna(cop_value) and 
                (cop_value < self.anomaly_thresholds['cop_abnormal_low'] or 
                 cop_value > self.anomaly_thresholds['cop_abnormal_high'])):
                anomalies['data_coupling_conflict'].append(idx)
                
        except Exception as e:
            logger.error(f"应用异常判定规则时出错: {e}")

    def fit_anomaly_classifier(self, historical_data, label_col, feature_cols=None):
        """训练异常分类器"""
        try:
            if feature_cols is None:
                feature_cols = [col for col in historical_data.columns 
                              if 'residual' in col or 'cop' in col]
            
            X = historical_data[feature_cols].fillna(0)
            y = historical_data[label_col]
            
            self.anomaly_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.anomaly_classifier.fit(X, y)
            self.anomaly_feature_cols = feature_cols
            
            logger.info("异常分类器训练完成")
            return True
        except Exception as e:
            logger.error(f"训练异常分类器时出错: {e}")
            return False

    def online_anomaly_detection(self, new_data_row):
        """实时异常检测"""
        try:
            result = {
                "physical_flags": {},
                "ml_anomaly_type": None,
                "residuals": {}
            }
            
            # 1. 计算残差
            data_df = pd.DataFrame([new_data_row])
            residuals = self.calculate_residuals(data_df)
            result["residuals"] = {k: v.iloc[0] for k, v in residuals.items()}
            
            # 2. 应用物理规则
            temp_residual = abs(result["residuals"].get('temp_supply_residual', 0))
            power_residual = abs(result["residuals"].get('power_residual', 0))
            cop_value = result["residuals"].get('cop_actual', np.nan)
            temp_diff = (new_data_row.get('temp_return', 0) - 
                        new_data_row.get('temp_supply', 0))
            
            # 检查各类异常
            if (temp_residual > self.anomaly_thresholds['temp_residual'] and 
                power_residual < 10):
                result["physical_flags"]["sensor_drift"] = True
            
            if (power_residual > self.anomaly_thresholds['power_residual'] and 
                not pd.isna(cop_value) and 
                cop_value < self.normal_ranges['cop_range'][0]):
                result["physical_flags"]["efficiency_issue"] = True
            
            # 3. 机器学习分类（如果有训练好的模型）
            if hasattr(self, 'anomaly_classifier'):
                features = pd.Series(index=self.anomaly_feature_cols)
                for col in self.anomaly_feature_cols:
                    if col in result["residuals"]:
                        features[col] = result["residuals"][col]
                    else:
                        features[col] = 0
                
                result["ml_anomaly_type"] = self.anomaly_classifier.predict([features])[0]
            
            return result
        except Exception as e:
            logger.error(f"在线异常检测时出错: {e}")
            return {}

    def validate_model(self, data):
        """验证物理模型"""
        try:
            # 准备数据
            X = self._prepare_features(data)
            y = data['cooling_load']  # 实际值
            
            # 获取预测值
            y_pred = self.predict(X)
            
            if y_pred is None or len(y_pred) == 0:
                logger.error("模型预测结果为空")
                return None
                
            # 绘制残差图
            residual_plot = self.visualizer.plot_residuals(
                data=data,
                predictions=y_pred,
                actual_values=y,
                save_path='reports/residual_plots.png'
            )
            
            if residual_plot is None:
                logger.error("残差图生成失败")
                
            # 返回验证结果
            return {
                'predictions': y_pred,
                'actual': y,
                'residuals': y - y_pred,
                'plot': residual_plot
            }
            
        except Exception as e:
            logger.error(f"模型验证过程出错: {str(e)}")
            return None

    def _prepare_features(self, data):
        """准备特征数据"""
        try:
            required_cols = ['temp_supply', 'temp_return', 'flow', 'power']
            
            # 检查必要的列是否存在
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"缺少必要的特征列: {missing_cols}")
                return None
                
            # 检查数据是否为空
            if data.empty:
                logger.error("输入数据为空")
                return None
                
            # 检查是否存在缺失值
            null_counts = data[required_cols].isnull().sum()
            if null_counts.any():
                logger.warning(f"特征中存在缺失值:\n{null_counts}")
                
            # 准备特征矩阵
            X = data[required_cols].copy()
            
            # 记录特征统计信息
            logger.info(f"特征数据形状: {X.shape}")
            logger.info(f"特征统计信息:\n{X.describe()}")
            
            return X
            
        except Exception as e:
            logger.error(f"特征准备过程出错: {str(e)}")
            return None

