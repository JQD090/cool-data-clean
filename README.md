# cool-data-clean
# cool-data-clean

## 项目简介

`cool-data-clean` 是一个面向工业冷却系统的智能数据预处理与异常分析工具箱。项目集成了数据质量分析、物理约束建模、异常检测、稳态识别、相似工况分析和自动化报告生成，旨在帮助工业用户高效完成冷却系统数据的清洗、质量评估与系统性诊断。

---

## 功能特性

- **数据加载与预处理**：支持 CSV、Excel 等格式数据的时序化读取，自动以时间戳为索引，异常与缺失值智能处理。
- **数据质量分析**：统计每个传感器/变量的缺失率、异常值率、恒定值率、噪声水平等，为后续清洗和决策提供依据。
- **物理约束建模**：内置能量守恒、COP 验证等物理约束规则，支持多类型异常（如传感器漂移、冷媒泄漏等）的自动判别。
- **多模态异常检测**：结合统计与物理模型，识别数据中的综合异常。
- **稳态识别与工况分析**：自动划分稳态区间，评估稳定性与数据完整性，辅助工况对比与追踪。
- **可视化与报告生成**：一键生成分析图表与 PDF 报告，自动输出关键建议，便于决策参考。

---

## 安装与环境

### 依赖环境

- Python >= 3.7
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- openpyxl
- 其他依赖详见 `requirements.txt`

### 安装

```bash
git clone https://github.com/JQD090/cool-data-clean.git
cd cool-data-clean
pip install -r requirements.txt
```

---

## 快速开始

1. **准备数据**  
   将你的冷却系统采集数据（含 `timestamp` 列）放在 `data/` 目录下，支持 `.csv` 或 `.xlsx`。

2. **运行主程序**

```bash
python -m colling_system.main  
```

- 首次运行会自动生成示例数据（如未检测到 `data/sample_data.csv`）。
- 日志将输出至 `cooling_system.log`。

3. **输出结果**

- 数据清洗/分析结果自动保存在内存和日志中。
- 报告自动生成 PDF 文件，默认存于项目目录。

---

## 主要模块说明

- `core/preprocessor.py`：数据预处理与质量分析。
- `core/physics_model.py`：冷却系统物理建模与多类型异常判别。
- `core/anomaly_detector.py`：多模态异常检测。
- `core/steady_state.py`：稳态区间识别与工况特征提取。
- `utils/visualization.py`：数据与分析结果可视化。
- `utils/report_generator.py`：自动化报告生成。  
- `main.py`：主流程入口。

---

## 配置与扩展

- 配置参数可在 `config/config.py` 或相关配置文件中自定义（如阈值、物理范围等）。
- 支持自定义数据源、物理模型与异常判别规则。

---

## 贡献与反馈

欢迎提交 Issue、PR 或建议！  
如遇到问题，请附带详细数据样本及运行日志，便于定位与修复。

---

## 许可证

本项目基于 MIT License 开源。

---
