#可视化工具
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import scipy.stats as stats 

logger = logging.getLogger(__name__)

class Visualizer:
    """可视化工具类"""
    
    def __init__(self):
        # 设置绘图样式
        plt.style.use('default')  # 使用默认样式
        sns.set_theme(style='ticks')  # 使用新版seaborn的主题设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        
        self.plot_configs = {
            'figsize': (12, 8),
            'dpi': 300,  
            'color_palette': ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C'],  # Nature风格配色
            'style': {
                'grid': True,
                'grid_alpha': 0.2,
                'title_size': 16,
                'label_size': 14,
                'tick_size': 12,
                'legend_size': 12,
                'annotation_size': 10
            },
            'font': {
                'family': 'Arial',
                'weight': 'bold'
            }
        }
        
        
        self.colors = {
            'normal': '#2878B5',     # 深蓝色
            'anomaly': '#C82423',    # 红色
            'steady': '#91CC75',     # 绿色
            'operation': '#FAC858',  # 橙色
            'grid': '#E6E6E6'        # 浅灰色网格
        }
        
        # 设置全局样式
        plt.rcParams.update({
            'font.family': ['SimHei', 'Arial'],  # 优先使用中文字体
            'font.size': 12,
            'axes.linewidth': 1.5,
            'axes.edgecolor': '#333333',
            'grid.color': '#E6E6E6',
            'grid.linestyle': '--',
            'axes.spines.top': True,
            'axes.spines.right': True,
            'axes.spines.left': True,
            'axes.spines.bottom': True
        })

    def generate_analysis_plots(self, data, analysis_results):
        """
        生成分析图表
        
        参数:
        - data: DataFrame 处理后的数据
        - analysis_results: dict 分析结果
        
        返回:
        - dict: 包含各类图表的字典
        """
        plots = {}
        
        try:
            # 1. 数据质量分析图
            if 'quality_report' in analysis_results:
                plots['data_quality'] = self._plot_data_quality(
                    analysis_results['quality_report']
                )
            
            # 2. 异常检测结果图
            if 'anomaly_detection' in analysis_results:
                plots['anomalies'] = self._plot_anomalies(
                    data, 
                    analysis_results['anomaly_detection']
                )
            
            # 3. 稳态识别结果图
            if 'steady_states' in analysis_results:
                plots['steady_states'] = self._plot_steady_states(
                    data, 
                    analysis_results['steady_states']
                )
            
            # 4. 相似工况分析图
            if 'operation_patterns' in analysis_results:
                plots['operations'] = self._plot_operation_patterns(
                    analysis_results['operation_patterns']
                )
            
            return plots
            
        except Exception as e:
            logger.error(f"生成分析图表时出错: {e}")
            return {}
    
    def _plot_data_quality(self, quality_report):
        """生成数据质量分析图"""
        try:
            if not quality_report or not isinstance(quality_report, dict):
                logger.error("数据质量报告为空或格式错误")
                return None

            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
            fig.suptitle('Data Quality Analysis', fontsize=16, y=1.02, fontweight='bold')

            # 1. 缺失率热力图
            ax1 = fig.add_subplot(gs[0, :])
            missing_data = pd.DataFrame({
                'sensor': list(quality_report.keys()),
                'missing_rate': [float(v['missing_rate']) for v in quality_report.values()]
            }).pivot_table(columns='sensor', values='missing_rate')

            sns.heatmap(missing_data, ax=ax1, cmap='coolwarm', annot=True, fmt='.1%', cbar_kws={'label': 'Missing Rate (%)'})
            ax1.set_title('Missing Data Distribution', pad=40)

            # 2. 噪声水平箱线图
            ax2 = fig.add_subplot(gs[1, 0])
            noise_data = pd.DataFrame({
                'Sensor': list(quality_report.keys()),
                'Noise Level': [float(v.get('noise_level', 0)) for v in quality_report.values()]
            })
            sns.boxplot(data=noise_data, x='Sensor', y='Noise Level', ax=ax2, color=self.colors['normal'])

            # 设置刻度和标签
            ax2.set_xticks(range(len(noise_data['Sensor'].unique())))
            ax2.set_xticklabels(noise_data['Sensor'].unique(), rotation=45, ha='right')
            ax2.set_title('Noise Level Distribution')

            # 3. 数据质量综合评分
            ax3 = fig.add_subplot(gs[1, 1])
            quality_scores = {
                sensor: 100 * (1 - v['missing_rate']) * (1 - v.get('noise_level', 0))
                for sensor, v in quality_report.items()
            }
            quality_df = pd.DataFrame(list(quality_scores.items()), columns=['Sensor', 'Quality Score'])
            sns.barplot(data=quality_df, x='Sensor', y='Quality Score', ax=ax3, color=self.colors['normal'])

            # 设置刻度和标签
            ax3.set_xticks(range(len(quality_df['Sensor'].unique())))
            ax3.set_xticklabels(quality_df['Sensor'].unique(), rotation=45, ha='right')
            ax3.set_title('Overall Data Quality Score')
            ax3.set_ylim(0, 100)

            # 添加数值标签
            for i, v in enumerate(quality_df['Quality Score']):
                ax3.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=self.plot_configs['style']['annotation_size'])

            plt.tight_layout()
            plt.close()  # 确保关闭图表
            return fig

        except Exception as e:
            logger.error(f"生成数据质量图表时出错: {e}")
            return plt.figure()  # 返回一个空图表
    
    def _plot_anomalies(self, data, anomaly_results):
        """生成异常检测结果图"""
        try:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])  # 调整比例
            fig.set_size_inches(18, 12)  # 增加图表尺寸
            fig.suptitle('异常检测分析', fontsize=16, y=1.02)
            
            # 1. 时间序列异常标记图
            ax1 = fig.add_subplot(gs[0, :])
            for col in ['temp_supply', 'power']:
                if col in data.columns:
                    ax1.plot(data.index, data[col], label=col, alpha=0.7, linewidth=1.5)
            
            # 修复异常点索引问题
            if 'combined_anomalies' in anomaly_results:
                anomaly_indices = anomaly_results['combined_anomalies']
                if anomaly_indices and isinstance(anomaly_indices, (list, np.ndarray)):
                    # 判断索引类型，若不是整数则转换为位置索引
                    if not isinstance(anomaly_indices[0], (int, np.integer)):
                        idx_map = {v: i for i, v in enumerate(data.index)}
                        valid_indices = [idx_map[idx] for idx in anomaly_indices if idx in idx_map]
                    else:
                        valid_indices = anomaly_indices
                    if valid_indices:
                        ax1.scatter(
                            [data.index[i] for i in valid_indices],
                            data['temp_supply'].iloc[valid_indices].values,
                            color=self.colors['anomaly'],
                            label='异常点',
                            marker='x',
                            s=100,
                            linewidth=2
                        )
            
            ax1.set_title('时间序列异常检测', fontsize=self.plot_configs['style']['title_size'])
            ax1.grid(True, alpha=self.plot_configs['style']['grid_alpha'])
            ax1.legend(fontsize=self.plot_configs['style']['legend_size'])
            
            # 2. 异常类型统计图
            ax2 = fig.add_subplot(gs[1, 0])
            anomaly_types = {}
            for key, values in anomaly_results.items():
                if isinstance(values, list):
                    anomaly_types[key] = len(values)
            
            # 修复barplot警告
            anomaly_df = pd.DataFrame({
                'Type': list(anomaly_types.keys()),
                'Count': list(anomaly_types.values())
            })
            sns.barplot(data=anomaly_df, x='Type', y='Count', ax=ax2, color=self.colors['anomaly'], hue=None)
            
            # 正确设置刻度标签
            ax2.set_xticks(range(len(anomaly_df['Type'])))
            ax2.set_xticklabels(anomaly_df['Type'], rotation=45, ha='right')
            ax2.set_title('异常类型统计', fontsize=self.plot_configs['style']['title_size'])
            
            # 添加数值标签
            for i, v in enumerate(anomaly_types.values()):
                ax2.text(i, v, str(v), ha='center', va='bottom')
            
            # 3. 异常时间分布热力图
            ax3 = fig.add_subplot(gs[1, 1])
            if 'combined_anomalies' in anomaly_results:
                # 直接用标签索引
                anomaly_times = pd.DatetimeIndex([data.index[i] if isinstance(i, int) else i for i in anomaly_results['combined_anomalies']])
                hour_day = pd.crosstab(anomaly_times.hour, anomaly_times.dayofweek)
                sns.heatmap(hour_day, ax=ax3, cmap='YlOrRd', cbar_kws={'label': '异常次数'})
                ax3.set_title('异常时间分布', fontsize=self.plot_configs['style']['title_size'])
                ax3.set_xlabel('星期', fontsize=self.plot_configs['style']['label_size'])
                ax3.set_ylabel('小时', fontsize=self.plot_configs['style']['label_size'])
            
            # 4. 异常模式分析
            ax4 = fig.add_subplot(gs[2, :])
            if 'physical_anomalies' in anomaly_results:
                physical_anomalies = anomaly_results['physical_anomalies']
                anomaly_patterns = pd.Series({k: len(v) for k, v in physical_anomalies.items()})
                sns.barplot(
                    data=pd.DataFrame({
                        'category': anomaly_patterns.index,
                        'value': anomaly_patterns.values
                    }),
                    x='category',
                    y='value',
                    ax=ax4,
                    color=self.plot_configs['color_palette'][0]  # 使用单一颜色
                )
                
                # 设置刻度和标签
                ax4.set_xticks(range(len(anomaly_patterns.index)))
                ax4.set_xticklabels(anomaly_patterns.index, rotation=45, ha='right')
                
                # 添加数值标签
                for i, v in enumerate(anomaly_patterns.values):
                    ax4.text(i, v, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.close()  # 确保关闭当前图表
            return fig
        
        except Exception as e:
            logger.error(f"生成异常检测图表时出错: {e}")
            return None
    
    def _plot_steady_states(self, data, steady_states):
        """生成稳态识别结果图"""
        try:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            fig.suptitle('Steady State Analysis', fontsize=16, y=0.95, fontweight='bold')
            
            # 1. 时间序列和稳态标记
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['temp_supply'], 
                    color=self.colors['normal'],
                    linewidth=1.5,
                    label='Supply Temperature')
            
            # 标记稳态区域
            for i, state in enumerate(steady_states):
                start = state['start']
                end = state['end']
                ax1.axvspan(start, end, 
                           color=self.colors['steady'],
                           alpha=0.2,
                           label=f'Steady State {i+1}' if i == 0 else None)
                
                # 添加持续时间标注
                duration = pd.Timedelta(end - start).total_seconds() / 3600
                ax1.text(start, ax1.get_ylim()[1], 
                        f'{duration:.1f}h',
                        fontsize=10,
                        va='bottom')
            
            ax1.set_title('Time Series with Steady States')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.2)
            
            # 2. 稳态持续时间分布
            ax2 = fig.add_subplot(gs[1])
            durations = [(pd.Timestamp(state['end']) - 
                         pd.Timestamp(state['start'])).total_seconds() / 3600 
                        for state in steady_states]
            
            sns.histplot(durations, ax=ax2, 
                        bins=30,
                        color=self.colors['steady'],
                        kde=True)
            ax2.set_title('Steady State Duration Distribution')
            ax2.set_xlabel('Duration (hours)')
            ax2.grid(True, alpha=0.2)
            
            # 3. 稳态特征分析
            ax3 = fig.add_subplot(gs[2])
            steady_features = self._analyze_steady_features(data, steady_states)
            # 修复boxplot颜色警告
            sns.boxplot(data=steady_features, ax=ax3, color=self.colors['steady'])
            
            # 正确设置刻度标签
            ax3.set_xticks(range(len(steady_features.columns)))
            ax3.set_xticklabels(steady_features.columns, rotation=45)
            ax3.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.close()  # 确保关闭当前图表
            return fig
            
        except Exception as e:
            logger.error(f"生成稳态识别图表时出错: {e}")
            return None
    
    def _analyze_steady_features(self, data, steady_states):
        """分析稳态特征分布"""
        features = []
        for state in steady_states:
            state_data = data.loc[state['start']:state['end']]
            features.append({
                'Temperature': state_data['temp_supply'].mean(),
                'Flow': state_data['flow'].mean() if 'flow' in state_data else None,
                'Power': state_data['power'].mean() if 'power' in state_data else None
            })
        return pd.DataFrame(features)

    def plot_residuals(self, data, predictions, actual_values, save_path=None):
        """绘制残差分析图"""
        try:
            # 数据验证
            if data is None or predictions is None or actual_values is None:
                logger.error("绘制残差图时数据为空")
                return None

            # 计算残差
            residuals = actual_values - predictions
            if len(residuals) == 0:
                logger.error("残差计算结果为空")
                return None

            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            plt.style.use('default')  # 重置样式

            # 1. 残差时间序列图
            axes[0, 0].plot(data.index, residuals, 'b.', alpha=0.7)
            axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[0, 0].set_title('残差时间序列', fontsize=self.plot_configs['style']['title_size'])
            axes[0, 0].set_xlabel('时间', fontsize=self.plot_configs['style']['label_size'])
            axes[0, 0].set_ylabel('残差', fontsize=self.plot_configs['style']['label_size'])
            axes[0, 0].grid(alpha=self.plot_configs['style']['grid_alpha'])

            # 2. 残差直方图
            sns.histplot(residuals, kde=True, ax=axes[0, 1], color=self.colors['normal'])
            axes[0, 1].set_title('残差分布', fontsize=self.plot_configs['style']['title_size'])
            axes[0, 1].set_xlabel('残差', fontsize=self.plot_configs['style']['label_size'])
            axes[0, 1].set_ylabel('频次', fontsize=self.plot_configs['style']['label_size'])
            axes[0, 1].grid(alpha=self.plot_configs['style']['grid_alpha'])

            # 3. Q-Q图
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q图', fontsize=self.plot_configs['style']['title_size'])
            axes[1, 0].grid(alpha=self.plot_configs['style']['grid_alpha'])

            # 4. 残差与预测值散点图
            axes[1, 1].scatter(predictions, residuals, alpha=0.5, color=self.colors['normal'])
            axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=1)
            axes[1, 1].set_title('残差 vs 预测值', fontsize=self.plot_configs['style']['title_size'])
            axes[1, 1].set_xlabel('预测值', fontsize=self.plot_configs['style']['label_size'])
            axes[1, 1].set_ylabel('残差', fontsize=self.plot_configs['style']['label_size'])
            axes[1, 1].grid(alpha=self.plot_configs['style']['grid_alpha'])

            # 调整布局
            plt.tight_layout()

            # 保存图形
            if save_path:
                plt.savefig(save_path, dpi=self.plot_configs['dpi'], bbox_inches='tight')
                logger.info(f"残差分析图已保存至: {save_path}")

            return fig

        except Exception as e:
            logger.error(f"绘制残差图时出错: {str(e)}")
            return None
    
    def _plot_operation_patterns(self, operation_patterns):
        """生成运行模式分析图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.plot_configs['figsize'])
            fig.suptitle('运行模式分析', fontsize=16, y=0.95)
            
            # 日内模式图
            if 'daily_patterns' in operation_patterns:
                daily_data = pd.DataFrame(operation_patterns['daily_patterns']).T
                sns.lineplot(data=daily_data, y='avg_load', x=daily_data.index,
                           ax=ax1, color=self.colors['normal'])
                ax1.set_title('日内负荷模式')
                ax1.set_xlabel('小时')
                ax1.set_ylabel('平均负荷')
                ax1.grid(True, alpha=0.2)
            
            # 周内模式图
            if 'weekly_patterns' in operation_patterns:
                weekly_data = pd.DataFrame(operation_patterns['weekly_patterns']).T
                # 确保索引是整数类型
                weekly_data.index = weekly_data.index.astype(int)
                weekly_data = weekly_data.sort_index()  # 确保正确排序
                
                sns.barplot(data=weekly_data, y='avg_load', x=weekly_data.index,
                          ax=ax2, color=self.colors['normal'], hue=None)
                ax2.set_xticks(range(len(weekly_data)))
                ax2.set_xticklabels(['周' + str(i) for i in weekly_data.index])
                
            plt.tight_layout()
            plt.close()  # 确保关闭当前图表
            return fig
            
        except Exception as e:
            logger.error(f"生成运行模式图表时出错: {e}")
            return None