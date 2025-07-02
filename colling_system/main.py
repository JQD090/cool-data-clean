import os
import logging
from datetime import datetime
from pathlib import Path
from config.config import Config
from core.preprocessor import DataPreprocessor
from core.physics_model import PhysicsConstraintModel
from core.anomaly_detector import MultiModalAnomalyDetector
from core.steady_state import SteadyStateIdentifier
from core.operation import SimilarOperationIdentifier
from utils.data_generator import DataGenerator 
from utils.visualization import Visualizer
from utils.report_generator import ReportGenerator
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cooling_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 配置绘图设置
plt.style.use('default')
sns.set_theme(style='ticks')  # 使用新版seaborn的主题设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

class CoolingSystemController:
    """制冷系统主控制器"""
    
    def __init__(self):
        """初始化主控制器"""
        self.config = Config()
        self._init_components()
        self._create_directories()
        
    def _init_components(self):
        """初始化各个组件"""
        try:
            self.preprocessor = DataPreprocessor(config=self.config)
            self.physics_model = PhysicsConstraintModel(config=self.config)
            self.anomaly_detector = MultiModalAnomalyDetector(config=self.config)
            self.steady_identifier = SteadyStateIdentifier(config=self.config)
            self.operation_analyzer = SimilarOperationIdentifier(config=self.config)
            self.visualizer = Visualizer()
            self.report_generator = ReportGenerator(config=self.config)
            
            self.data = None
            self.processed_data = None
            self.analysis_results = {}
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.DATA_DIR,
            self.config.REPORT_DIR,
            self.config.TEMPLATE_DIR,
            'logs'
        ]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def load_and_process_data(self, file_path):
        """加载并预处理数据"""
        logger.info("开始加载数据...")
        try:
            # 加载数据
            self.data = self.preprocessor.load_data(file_path)
            if self.data is None:
                return False
                
            # 数据质量分析
            logger.info("进行数据质量分析...")
            quality_report = self.preprocessor.check_data_quality(self.data)
            self.analysis_results['quality_report'] = quality_report
            
            # 数据清洗
            logger.info("进行数据清洗...")
            self.processed_data = self.preprocessor.clean_data(self.data, quality_report)
            
            logger.info(f"数据处理完成。原始数据: {self.data.shape}, 清洗后数据: {self.processed_data.shape}")
            return True
            
        except Exception as e:
            logger.error(f"数据处理过程出错: {e}")
            return False
    
    def run_comprehensive_analysis(self):
        """运行综合分析"""
        if self.processed_data is None:
            logger.error("请先加载数据")
            return False
            
        try:
            logger.info("开始综合分析...")
            
            # 1. 物理约束验证
            logger.info("进行物理约束验证...")
            physical_results = self.physics_model.detect_physical_anomalies(self.processed_data)
            self.analysis_results['physical_analysis'] = physical_results
            
            # 2. 异常检测
            logger.info("进行异常检测...")
            anomaly_results = self.anomaly_detector.detect_comprehensive_anomalies(self.processed_data)
            self.analysis_results['anomaly_detection'] = anomaly_results
            
            # 3. 稳态识别
            logger.info("进行稳态识别...")
            steady_states = self.steady_identifier.identify_steady_states(self.processed_data)
            self.analysis_results['steady_states'] = steady_states
            
            # 4. 相似工况分析
            logger.info("进行相似工况分析...")
            operation_patterns = self.operation_analyzer.analyze_similar_operations(self.processed_data)
            self.analysis_results['operation_patterns'] = operation_patterns
            
            logger.info("综合分析完成")
            return True
            
        except Exception as e:
            logger.error(f"分析过程出错: {e}")
            return False
    
    def generate_report(self):
        """生成分析报告"""
        try:
            logger.info("开始生成分析报告...")
            
            # 生成图表
            plots = self.visualizer.generate_analysis_plots(
                self.processed_data,
                self.analysis_results
            )
            
            # 生成建议
            suggestions = self._generate_suggestions()
            
            # 直接生成PDF报告
            pdf_path = self.report_generator.generate_report(
                self.analysis_results,
                plots,
                suggestions
            )
            
            if pdf_path:
                logger.info(f"报告已保存至: {pdf_path}")
                return pdf_path
            return None
            
        except Exception as e:
            logger.error(f"报告生成过程出错: {e}")
            return None
    
    def _generate_suggestions(self):
        """根据分析结果生成建议"""
        suggestions = []
        
        # 根据数据质量生成建议
        if 'quality_report' in self.analysis_results:
            for sensor, metrics in self.analysis_results['quality_report'].items():
                if metrics['missing_rate'] > 0.1:
                    suggestions.append(f"传感器 {sensor} 数据缺失率较高，建议检查传感器状态")
                if metrics['noise_level'] > 0.2:
                    suggestions.append(f"传感器 {sensor} 噪声水平较高，建议进行校准")
        
        # 根据物理约束验证结果生成建议
        if 'physical_analysis' in self.analysis_results:
            physics_results = self.analysis_results['physical_analysis']
            if physics_results.get('efficiency_issues', False):
                suggestions.append("系统效率低于预期，建议检查设备运行状态")
                
        # 根据异常检测结果生成建议
        if 'anomaly_detection' in self.analysis_results:
            anomaly_results = self.analysis_results['anomaly_detection']
            if len(anomaly_results.get('combined_anomalies', [])) > 0:
                suggestions.append("检测到系统异常，建议进行深入诊断")
        
        return suggestions

def main():
    """主程序入口"""
    try:
        logger.info("系统初始化...")
        controller = CoolingSystemController()
        
        # 生成示例数据（如果需要）
        sample_data_path = os.path.join('data', 'sample_data.csv')
        if not os.path.exists(sample_data_path):
            logger.info("生成示例数据...")
            data_generator = DataGenerator()  # 创建数据生成器实例
            data_generator.generate_sample_data(output_path=sample_data_path)  # 生成示例数据
        
        # 加载和处理数据
        if controller.load_and_process_data(sample_data_path):
            # 运行分析
            if controller.run_comprehensive_analysis():
                # 生成报告
                report_path = controller.generate_report()
                if report_path:
                    logger.info("分析完成，报告已生成")
                    return True
        
        logger.error("处理过程未完成")
        return False
        
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        return False

if __name__ == "__main__":
    main()