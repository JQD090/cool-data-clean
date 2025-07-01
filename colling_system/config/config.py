#配置文件
class Config:
    """系统配置"""
    # 物理参数
    WATER_DENSITY = 1000  # kg/m³
    WATER_SPECIFIC_HEAT = 4.2  # kJ/(kg·℃)
    
    # 文件路径
    DATA_DIR = "data"
    REPORT_DIR = "reports"
    TEMPLATE_DIR = "templates"
    TEMPLATE_FILE = "report_template.html"
    
    # 分析参数
    STEADY_STATE_WINDOW = 30
    ANOMALY_THRESHOLDS = {
        'residual': 2.0,
        'cop_range': (3.0, 5.0)
    }
    
    # 传感器配置
    SENSOR_TYPES = {
        'temperature': {'unit': '℃', 'type': 'analog', 'frequency': '1-5min'},
        'power': {'unit': 'kW', 'type': 'digital', 'frequency': '1min'},
        'flow': {'unit': 'm³/h', 'type': 'digital', 'frequency': '1min'},
        'pressure': {'unit': 'MPa', 'type': 'analog', 'frequency': '5min'},
        'valve_opening': {'unit': '%', 'type': 'control', 'frequency': 'variable'},
        'environment': {'unit': '℃/RH%', 'type': 'external', 'frequency': '15min-1h'}
    }

    def __init__(self):
        self.TEMPLATE_DIR = "templates"
        self.TEMPLATE_FILE = "report_template.html"