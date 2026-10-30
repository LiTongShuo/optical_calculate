# optical_calculate

中文 | [English](README.en.md)

用于光学超振荡透镜研究的 Python 脚本集，包含随机环带半径数据生成、衍射积分计算、光学参数评估、随机仿真以及 GA/NSGA-II 优化实验。

## 项目背景与目标

本项目意图寻找一种特定的超振荡透镜结构。超振荡透镜通过对透镜结构进行精细设计，在远场或焦场中产生局部尺度小于常规衍射限制的高集中光场，从而实现更小焦斑或更强局部能量约束。

在本项目中，超振荡透镜结构由一组环带半径参数 `rm` 描述。因此，寻找合适的超振荡透镜结构，本质上就是在给定物理约束和加工精度下，寻找一组满足目标光场性能的特定数列。

研究目标是在太赫兹频率下获得弧矢面光场分布集中、焦斑较小，同时子午面具有较长焦距和较长焦深的透镜结构。该方向计划用于太赫兹 CT 成像等需要高分辨率和长焦深光场控制的应用。

## 功能概览

- 生成 RM 环带半径训练/验证/测试数据集
- 基于衍射积分计算焦距、FWHM、压缩比和旁瓣比
- 执行高精度随机仿真并输出 CSV 与图表
- 使用遗传算法和 NSGA-II 搜索候选环带半径序列
- 绘制典型 RM 序列的二维传播强度图

## 环境安装

建议使用 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 快速开始

生成一个小样本数据集：

```powershell
python scripts/generate_rm_dataset.py --num-samples 100 --output-dir data/rm_dataset
```

计算生成数据集的光学参数：

```powershell
python scripts/compute_optical_params.py --input data/rm_dataset/train_dataset.csv --output outputs/optical_params_results.csv --processes 2
```

运行小规模随机仿真：

```powershell
python scripts/simulation.py --num-samples 100 --output-dir outputs
```

绘制 NSGA-II 示例传播图：

```powershell
python scripts/optics_nsga2.py --mode plot --output-dir outputs
```

## 脚本说明

- `scripts/generate_rm_dataset.py`：生成基础 RM 数据集。
- `scripts/generate_rm_dataset_v2.py`：生成带 `r1=0` 样本比例控制的数据集。
- `scripts/compute_optical_params.py`：并行计算数据集中每组 RM 的光学参数。
- `scripts/diffraction_integral.py`：单组或多组 RM 的衍射积分分析。
- `scripts/diffraction_integral_v2.py`：带峰值/FWHM 分析的衍射积分分析版本。
- `scripts/simulation.py`：高精度随机仿真，筛选并保存满足条件的结果。
- `scripts/optics_ga.py`：遗传算法优化实验。
- `scripts/optics_nsga2.py`：NSGA-II 多目标优化与传播图绘制。
- `scripts/dna.py`：差分进化优化实验。

## 数据与结果

完整数据集、模型文件和生成结果不随仓库提交。请通过脚本重新生成数据；输出文件默认写入 `outputs/`。

## License

This project is licensed under the MIT License.
