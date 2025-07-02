#报告生成工具
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import io
import textwrap

logger = logging.getLogger(__name__)

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, config=None):
        self.config = config
        self.output_dir = config.REPORT_DIR if config else "reports"
        
        # 检查 matplotlib 版本
        import matplotlib
        self.mpl_version = matplotlib.__version__
        logger.info(f"使用 matplotlib 版本: {self.mpl_version}")
        
        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_report(self, analysis_results, plots, suggestions):
        """直接生成PDF报告"""
        try:
            if not isinstance(analysis_results, dict):
                raise ValueError("analysis_results 必须是字典类型")
            if not isinstance(plots, dict):
                raise ValueError("plots 必须是字典类型")
            if not isinstance(suggestions, list):
                raise ValueError("suggestions 必须是列表类型")
            
            logger.info("开始生成PDF报告...")
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 生成PDF文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(self.output_dir, f'analysis_report_{timestamp}.pdf')
            
            # 使用PdfPages直接生成PDF
            with PdfPages(pdf_path) as pdf:
                # 1. 添加封面页
                self._add_cover_page(pdf)
                
                # 2. 添加其他页面
                for plot_key in ['data_quality', 'anomalies', 'steady_states', 'operations']:
                    if plot_key in plots and plots[plot_key] is not None:
                        try:
                            pdf.savefig(plots[plot_key])
                        except Exception as e:
                            logger.error(f"保存 {plot_key} 图表时出错: {e}")
                        finally:
                            plt.close()  # 确保清理资源
                
                # 3. 添加建议页
                self._add_suggestions_page(pdf, suggestions)
            
            logger.info(f"成功添加 {len(plots)} 个图表")
            logger.info(f"成功添加 {len(suggestions)} 条建议")
            logger.info(f"PDF报告已生成: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"生成PDF报告时出错: {e}")
            plt.close('all')  # 确保清理所有图形资源
            return None
    
    def _add_cover_page(self, pdf):
        """添加封面页"""
        try:
            # 不使用 with 语句
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # 添加标题
            ax.text(0.5, 0.7, '制冷系统运行分析报告', 
                    ha='center', va='center', 
                    fontsize=24, fontweight='bold')
            
            # 添加生成时间
            ax.text(0.5, 0.5, f'生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    ha='center', va='center',
                    fontsize=12)
            
            pdf.savefig(fig)
            plt.close(fig)  # 显式关闭图形
        except Exception as e:
            logger.error(f"生成封面页时出错: {e}")
            plt.close()  # 确保清理资源
    
    def _wrap_text(self, text, width=50):
        """文本自动换行"""
        return textwrap.fill(text, width=width)

    def _add_suggestions_page(self, pdf, suggestions):
        """添加建议页"""
        try:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')
            
            # 添加页面边距
            margin = 0.1
            text_width = 1 - 2 * margin
            
            # 添加标题
            ax.text(0.5, 1 - margin, '运行建议', 
                    ha='center', va='top',
                    fontsize=16, fontweight='bold')
            
            # 添加建议内容
            y_pos = 0.85
            for i, suggestion in enumerate(suggestions, 1):
                wrapped_text = self._wrap_text(suggestion)
                ax.text(0.1, y_pos, f"{i}. {wrapped_text}",
                       ha='left', va='top',
                       fontsize=10)
                y_pos -= 0.1 * (1 + wrapped_text.count('\n'))
            
            pdf.savefig(fig)
            plt.close(fig)  # 显式关闭图形
        except Exception as e:
            logger.error(f"生成建议页时出错: {e}")
            plt.close()  # 确保清理资源