#准稳态识别模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class SteadyStateIdentifier:
    """稳态识别模块"""
    
    def __init__(self, config=None):
        """初始化稳态识别器
        
        参数:
            config: 配置对象，可选
        """
        self.config = config
        
        # 从配置加载参数，如果没有配置则使用默认值
        if config:
            self.window_size = getattr(config, 'STEADY_STATE_WINDOW', 30)
            self.stability_thresholds = getattr(config, 'STABILITY_THRESHOLDS', {
                'temperature': 0.05,
                'flow': 0.05,
                'power': 0.05,
                'pressure': 0.05
            })
        else:
            # 默认参数
            self.window_size = 30
            self.stability_thresholds = {
                'temperature': 0.05,  # 温度变化阈值（相对值）
                'flow': 0.05,        # 流量变化阈值（相对值）
                'power': 0.05,       # 功率变化阈值（相对值）
                'pressure': 0.05     # 压力变化阈值（相对值）
            }
        
        # 稳态判定参数
        self.steady_state_params = {
            'min_duration': 60,      # 最小稳态持续时间（数据点）
            'max_gap': 5,            # 最大允许的数据间断（数据点）
            'confidence_threshold': 0.9  # 稳态置信度阈值
        }

    
    def identify_steady_states(self, data, key_variables=None):
        """识别准稳态时段"""
        if key_variables is None:
            # 自动检测所有支持的变量
            key_variables = [v for v in ['temp_supply', 'flow', 'power', 'pressure'] if v in data.columns]
        
        # 过滤存在的变量
        available_vars = [var for var in key_variables if var in data.columns]
        if not available_vars:
            return []
        
        steady_states = []
        
        # 计算变化率
        change_rates = {}
        for var in available_vars:
            # 计算滑动窗口内的变化率
            rolling_std = data[var].rolling(window=self.window_size, min_periods=10).std()
            rolling_mean = data[var].rolling(window=self.window_size, min_periods=10).mean()
            change_rates[var] = rolling_std / (rolling_mean.abs() + 1e-6)
        
        # 综合判断稳态
        is_steady = pd.Series(True, index=data.index)
        for var, rates in change_rates.items():
            threshold = self._get_threshold(var)
            is_steady &= (rates < threshold) | rates.isna()
        
        # 提取连续稳态段
        steady_segments = self._extract_continuous_segments(is_steady)
        
        # 过滤短时稳态（至少60个数据点）
        min_points = 60
        steady_states = [
            seg for seg in steady_segments 
            if len(data[seg['start']:seg['end']]) >= min_points
        ]
        
        return steady_states
    
    def _get_threshold(self, variable):
        """获取变量的稳定性阈值"""
        for key, threshold in self.stability_thresholds.items():
            if key in variable.lower():
                return threshold
        return 0.05  # 默认阈值
    
    def _extract_continuous_segments(self, is_steady):
        """提取连续的稳态段"""
        segments = []
        in_segment = False
        start_idx = None
        
        for i, (idx, steady) in enumerate(is_steady.items()):
            if steady and not in_segment:
                # 开始新的稳态段
                in_segment = True
                start_idx = idx
            elif not steady and in_segment:
                # 结束当前稳态段
                in_segment = False
                if start_idx is not None:
                    segments.append({
                        'start': start_idx,
                        'end': is_steady.index[max(0, i-1)],
                        'duration': is_steady.index[max(0, i-1)] - start_idx
                    })
        
        # 处理最后一个段
        if in_segment and start_idx is not None:
            segments.append({
                'start': start_idx,
                'end': is_steady.index[-1],
                'duration': is_steady.index[-1] - start_idx
            })
        
        return segments
    
    def analyze_steady_state_quality(self, data, steady_states):
        """分析稳态质量"""
        quality_metrics = []
        
        for state in steady_states:
            try:
                segment_data = data[state['start']:state['end']]
                
                if len(segment_data) == 0:
                    continue
                
                metrics = {
                    'start': state['start'],
                    'end': state['end'],
                    'duration_minutes': len(segment_data) * 5,  # 假设5分钟间隔
                    'mean_values': {},
                    'stability_scores': {},
                    'data_completeness': 1 - segment_data.isna().sum().sum() / segment_data.size
                }
                
                # 计算各变量的均值和稳定性得分
                for col in segment_data.columns:
                    if segment_data[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                        col_data = segment_data[col].dropna()
                        if len(col_data) > 0:
                            metrics['mean_values'][col] = col_data.mean()
                            metrics['stability_scores'][col] = 1 - (
                                col_data.std() / (col_data.mean() + 1e-6)
                            )
                
                # 综合稳定性得分
                if metrics['stability_scores']:
                    metrics['overall_stability'] = np.mean(list(metrics['stability_scores'].values()))
                else:
                    metrics['overall_stability'] = 0
                
                quality_metrics.append(metrics)
            except Exception as e:
                print(f"分析稳态质量时出错: {e}")
                continue
        
        return quality_metrics

    def plot_steady_states(self, data, steady_states, variables=None):
        """
        可视化稳态段
        参数:
        - data: DataFrame，原始数据
        - steady_states: list，稳态段列表
        - variables: list，要显示的变量列表，默认显示所有关键变量
        """
        if variables is None:
            variables = ['temp_supply', 'flow', 'power']
        
        variables = [var for var in variables if var in data.columns]
        if not variables:
            print("没有可用的变量进行可视化")
            return
        
        # 创建子图
        fig, axes = plt.subplots(len(variables), 1, figsize=(15, 4*len(variables)))
        if len(variables) == 1:
            axes = [axes]
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(steady_states)))
        
        for ax, var in zip(axes, variables):
            # 绘制原始数据
            data[var].plot(ax=ax, alpha=0.5, label='原始数据', color='gray')
            
            # 标记稳态段
            for i, state in enumerate(steady_states):
                steady_data = data.loc[state['start']:state['end'], var]
                ax.fill_between(steady_data.index, 
                              steady_data.min(), 
                              steady_data.max(),
                              color=colors[i], alpha=0.3)
                ax.plot(steady_data.index, steady_data, 
                       color=colors[i], linewidth=2,
                       label=f'稳态段 {i+1}')
            
            ax.set_title(f'{var} 稳态识别结果')
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def adjust_thresholds(self, data, history_window='7D'):
        """
        根据历史数据动态调整稳态判定阈值
        参数:
        - data: DataFrame，历史数据
        - history_window: str，历史数据窗口大小
        """
        try:
            # 获取历史数据窗口
            end_time = data.index[-1]
            start_time = end_time - pd.Timedelta(history_window)
            historical_data = data[data.index >= start_time]
            
            new_thresholds = {}
            
            for var_type, threshold in self.stability_thresholds.items():
                matching_cols = [col for col in data.columns 
                               if var_type.lower() in col.lower()]
                
                if not matching_cols:
                    continue
                
                # 计算变化率的分布
                rates = []
                for col in matching_cols:
                    series = historical_data[col]
                    rolling_std = series.rolling(window=self.window_size, 
                                              min_periods=10).std()
                    rolling_mean = series.rolling(window=self.window_size, 
                                               min_periods=10).mean()
                    rates.extend((rolling_std / (rolling_mean.abs() + 1e-6)).dropna())
                
                if rates:
                    # 使用分位数作为新阈值
                    new_threshold = np.percentile(rates, 90)  # 90分位数
                    new_thresholds[var_type] = min(max(new_threshold, 
                                                     threshold * 0.5),  # 下限
                                                 threshold * 2)  # 上限
            
            # 更新阈值
            self.stability_thresholds.update(new_thresholds)
            print("已更新稳态判定阈值:", self.stability_thresholds)
            
        except Exception as e:
            print(f"调整阈值时出错: {e}")

    def predict_steady_states(self, data, prediction_horizon='1D'):
        """
        预测未来可能的稳态时段
        参数:
        - data: DataFrame，历史数据
        - prediction_horizon: str，预测时间范围
        返回:
        - predicted_states: list，预测的稳态时段列表
        """
        try:
            # 1. 分析历史稳态模式
            steady_states = self.identify_steady_states(data)
            if not steady_states:
                return []
            
            # 2. 提取稳态特征
            steady_patterns = []
            for state in steady_states:
                start_time = pd.to_datetime(state['start'])
                duration = (state['end'] - state['start']).total_seconds() / 3600  # 小时
                steady_patterns.append({
                    'hour_of_day': start_time.hour,
                    'day_of_week': start_time.dayofweek,
                    'duration': duration
                })
            
            # 3. 统计稳态出现规律
            hour_probs = pd.DataFrame(steady_patterns)['hour_of_day'].value_counts(normalize=True)
            day_probs = pd.DataFrame(steady_patterns)['day_of_week'].value_counts(normalize=True)
            avg_duration = pd.DataFrame(steady_patterns)['duration'].mean()
            
            # 4. 生成预测时间范围
            end_time = data.index[-1]
            future_times = pd.date_range(
                start=end_time,
                end=end_time + pd.Timedelta(prediction_horizon),
                freq='1H'
            )
            
            # 5. 预测稳态段
            predicted_states = []
            current_state = None
            
            for time in future_times:
                hour_prob = hour_probs.get(time.hour, 0)
                day_prob = day_probs.get(time.dayofweek, 0)
                
                # 综合概率
                steady_prob = (hour_prob + day_prob) / 2
                
                if steady_prob > 0.3:  # 概率阈值
                    if current_state is None:
                        current_state = {
                            'start': time,
                            'probability': steady_prob
                        }
                elif current_state is not None:
                    current_state['end'] = time
                    current_state['duration'] = \
                        (current_state['end'] - current_state['start']).total_seconds() / 3600
                    if current_state['duration'] >= 2:  # 最小持续时间
                        predicted_states.append(current_state)
                    current_state = None
            
            return predicted_states
            
        except Exception as e:
            print(f"预测稳态段时出错: {e}")
            return []