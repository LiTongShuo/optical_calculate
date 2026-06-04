# optics_ga.py (假设包含了 simulation.py 中的 CONFIG 和 OpticalParamsCalculator)

import numpy as np
import pandas as pd
import multiprocessing as mp
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import time
import warnings
import argparse
import os
# 引入 simulation.py 中的所有必要模块
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import j0
from scipy.integrate import simpson
# 忽略警告
warnings.filterwarnings('ignore')

# ================= 原始 CONFIG 和 OpticalParamsCalculator 类 =================
# ⚠️ 注意：这里应完整复制 simulation.py 中的 CONFIG 和 OpticalParamsCalculator 类
# 为了代码可运行，我在此省略，请确保它们存在于本文件或可导入。
# ... (CONFIG 字典)
# ... (OpticalParamsCalculator 类)
CONFIG = {
    'wavelength': 1.08,           # 波长 (mm)
    'r_max': 25.0,                # 最大半径 (mm)
    'rm_count': 20,               # 固定 rm 数量
    'rm_precision': 0.01,         # 加工精度/模拟精度
    'num_integration_points': 1000, # 积分精度 (提高以保证物理准确性)
    
    # 焦距搜索设置
    'z_range': (10, 100),         
    'num_z_points': 150,          
    
    # 横向场设置 (关键改进点)
    'r_prime_range': (-3.0, 3.0), # 缩小范围专注中心，减少计算量
    'num_r_prime_points': 5000,   # 采样点数 5000 -> 步长 0.0012mm (极高精度)
}

class OpticalParamsCalculator:
    def __init__(self, wavelength: float, r_max: float, num_integration_points: int):
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        self.r_max = r_max
        self.r_integration = np.linspace(0, self.r_max, num_integration_points)
    
    def compute_T_values(self, rm_values: list) -> np.ndarray:
        T_arr = np.ones_like(self.r_integration)
        for i in range(0, len(rm_values) - 1, 2):
            if i + 1 < len(rm_values):
                mask = (self.r_integration >= rm_values[i]) & (self.r_integration <= rm_values[i+1])
                T_arr[mask] = -1
        return T_arr
    
    def compute_integrand(self, r_prime: float, z: float, T_values: np.ndarray) -> np.ndarray:
        k = 2 * np.pi / self.wavelength
        r_squared = self.r_integration**2
        T_r = T_values * self.r_integration
        sqrt_term = np.sqrt(r_prime**2 + r_squared + z**2)
        phase = np.exp(1j * k * sqrt_term)
        bessel_arg = (k * r_prime * self.r_integration) / sqrt_term
        bessel_term = j0(bessel_arg)
        return (self.k0 / (1j * z)) * T_r * phase * bessel_term
    
    def get_interpolated_fwhm(self, r_axis, intensity, max_val):
        """
        使用线性插值计算精确的 FWHM
        """
        half_max = max_val / 2.0
        peak_idx = np.argmax(intensity)
        
        # --- 寻找右侧交叉点 ---
        # 找到峰值右侧第一个低于半高值的索引
        right_indices = np.where(intensity[peak_idx:] < half_max)[0]
        if len(right_indices) == 0: return None # 没降到半高以下
        
        idx2 = peak_idx + right_indices[0] # 低于半高的点
        idx1 = idx2 - 1                    # 还在半高之上的点
        
        # 线性插值公式: x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
        r_right = r_axis[idx1] + (half_max - intensity[idx1]) * \
                  (r_axis[idx2] - r_axis[idx1]) / (intensity[idx2] - intensity[idx1])
        
        # --- 寻找左侧交叉点 ---
        # 找到峰值左侧第一个低于半高值的索引 (倒序搜索)
        left_indices = np.where(intensity[:peak_idx] < half_max)[0]
        if len(left_indices) == 0: return None
        
        idx1_L = left_indices[-1] # 低于半高的点 (左侧最靠近峰值的)
        idx2_L = idx1_L + 1       # 还在半高之上的点
        
        r_left = r_axis[idx1_L] + (half_max - intensity[idx1_L]) * \
                 (r_axis[idx2_L] - r_axis[idx1_L]) / (intensity[idx2_L] - intensity[idx1_L])
                 
        return abs(r_right - r_left)

    def calculate_all_params(self, rm_values: list) -> dict:
        try:
            T_values = self.compute_T_values(rm_values)
            
            # 1. 寻找焦距 (f)
            z_values = np.linspace(CONFIG['z_range'][0], CONFIG['z_range'][1], CONFIG['num_z_points'])
            intensities_z = []
            for z in z_values:
                integrand = self.compute_integrand(0.0, z, T_values)
                intensities_z.append(np.abs(simpson(integrand, self.r_integration))**2)
            
            focal_idx = np.argmax(intensities_z)
            focal_length = z_values[focal_idx]
            max_intensity = intensities_z[focal_idx]

            # 2. 计算高分辨率横向分布
            r_prime_values = np.linspace(CONFIG['r_prime_range'][0], CONFIG['r_prime_range'][1], CONFIG['num_r_prime_points'])
            intensities_r = []
            
            # 这里的循环是性能瓶颈，但为了精度必须保留
            for r_p in r_prime_values:
                integrand = self.compute_integrand(r_p, focal_length, T_values)
                intensities_r.append(np.abs(simpson(integrand, self.r_integration))**2)
            
            intensities_r = np.array(intensities_r)
            Im = np.max(intensities_r)
            
            # 3. 精确计算 FWHM
            Rm = self.get_interpolated_fwhm(r_prime_values, intensities_r, Im)
            if Rm is None: return {'success': False, 'error': 'FWHM calculation failed'}
            
            compression_ratio = Rm / self.wavelength
            
            # 4. 计算旁瓣比
            # 排除主瓣区域 (比如 FWHM 的 1.5 倍范围)
            main_lobe_width = Rm * 1.5
            mask_sidelobe = np.abs(r_prime_values) > (main_lobe_width / 2.0)
            
            if np.any(mask_sidelobe):
                sidelobe_intensity = np.max(intensities_r[mask_sidelobe])
                sidelobe_ratio = sidelobe_intensity / Im
            else:
                sidelobe_ratio = 0.0

            return {
                'focal_length': float(focal_length),
                'FWHM': float(Rm),
                'compression_ratio': float(compression_ratio),
                'sidelobe_ratio': float(sidelobe_ratio),
                'max_intensity': float(max_intensity),
                'success': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# 示例: 最小化 FWHM, 限制旁瓣比 (SR < 0.2)
CONFIG = {
    'wavelength': 1.08,           # 波长 (mm)
    'r_max': 25.0,                # 最大半径 (mm)
    'rm_count': 20,               # 固定 rm 数量 (遗传算法的基因数量)
    'rm_precision': 0.01,         # 加工精度/模拟精度 (个体取值范围约束)
    'num_integration_points': 1000, # 积分精度
    
    # 焦距搜索设置
    'z_range': (10, 100),         
    'num_z_points': 150,          
    
    # 横向场设置
    'r_prime_range': (-3.0, 3.0), 
    'num_r_prime_points': 5000,   
}

# 辅助函数：计算适应度
def evaluate_rm_sequence(rm_values: list):
    """
    计算单个 rm 序列的适应度 (Fitness)。
    目标: 最小化 FWHM, 惩罚 SR > SR_limit 的情况。
    """
    SR_limit = 0.2  # 旁瓣比上限
    Penalty_Factor = 100.0 * CONFIG['wavelength'] # 惩罚系数
    
    # 排序并去重 (保持序列物理有效)
    rm_values = np.sort(np.unique(rm_values)).tolist()
    if len(rm_values) % 2 != 0: 
        # 如果长度是奇数，可能无法进行计算，给一个极大的惩罚值
        return (1000 * CONFIG['wavelength'],)

    # 初始化计算器
    calculator = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])
    res = calculator.calculate_all_params(rm_values)

    if not res['success']:
        # 计算失败 (例如 FWHM 没找到)，给一个极大的惩罚值
        return (1000 * CONFIG['wavelength'],) 
    
    fwhm = res['FWHM']
    sidelobe_ratio = res['sidelobe_ratio']
    
    # 适应度 = FWHM + 旁瓣比惩罚项
    penalty = 0.0
    if sidelobe_ratio > SR_limit:
        penalty = Penalty_Factor * (sidelobe_ratio - SR_limit)
        
    # 确保 FWHM 不为负 (虽然物理上不会)
    fitness = max(fwhm, 0) + penalty
    
    # DEAP 最小化适应度 (元组形式)
    return (fitness,)

# 辅助函数：并行评估
def parallel_evaluate(individuals, pool):
    """
    使用多进程并行评估 DEAP 中的所有个体。
    """
    # 提取个体基因
    tasks = [list(ind) for ind in individuals]
    # 使用 imap_unordered 以便在 tqdm 中显示进度
    results = list(tqdm(pool.imap_unordered(evaluate_rm_sequence, tasks), 
                        total=len(individuals), desc="评估代数"))
    return results

# ================= 遗传算法配置 =================

# 1. 设置优化目标: 最小化 ('weights': (-1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 2. 个体 (Individual): 列表形式，带上 Fitness 属性
creator.create("Individual", list, fitness=creator.FitnessMin)

# 初始化工具箱
toolbox = base.Toolbox()

# 基因生成器: 随机浮点数 (0 到 r_max 之间)
# 注意：实际应用中，我们会确保 rm_values 列表的长度是偶数，
# 但在 DEAP 中，我们只生成 rm_count 个基因，并在评估时处理长度。
toolbox.register("attr_float", np.random.uniform, 0.0, CONFIG['r_max'])

# 个体初始化: 生成 rm_count 个浮点数
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=CONFIG['rm_count'])

# 种群初始化
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# GA 算子注册
# 评估函数 (由我们自定义的并行评估函数调用)
toolbox.register("evaluate", evaluate_rm_sequence) 

# 选择: 锦标赛选择
toolbox.register("select", tools.selTournament, tournsize=3) 

# 交叉: 两点交叉
toolbox.register("mate", tools.cxTwoPoint)

# 变异: 高斯变异 (对每个基因进行扰动)
# mu: 均值, sigma: 标准差, indpb: 变异概率
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.0, indpb=0.1)

# ================= 主程序 (GA 运行) =================
def run_ga_optimization(pop_size=100, num_generations=50, cxpb=0.7, mutpb=0.2, output_dir="outputs"):
    
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print("=== 超振荡透镜 GA 优化开始 ===")
    print(f"种群大小: {pop_size} | 迭代次数: {num_generations} | 基因数: {CONFIG['rm_count']}")
    
    pool_size = max(1, mp.cpu_count() - 2) 
    print(f"正在计算... (使用 {pool_size} 核心进行评估)")
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    # 记录统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    logbook = tools.Logbook()
    
    # 精英保留，记录最好的个体
    hof = tools.HallOfFame(1)

    # 多进程评估
    with mp.Pool(processes=pool_size) as pool:
        
        # 第一次评估
        fitnesses = parallel_evaluate(pop, pool)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, **record)
        print(logbook.stream)

        # 迭代开始
        for gen in range(1, num_generations + 1):
            
            # 1. 选择下一代个体
            # 使用 NSGA-II 的选择方式可以更好地处理约束和多目标，这里用 selTournament 简化
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            # 2. 应用交叉和变异
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < mutpb:
                    # 确保变异后的值在 [0, r_max] 范围内
                    toolbox.mutate(mutant)
                    mutant[:] = [max(0.0, min(CONFIG['r_max'], x)) for x in mutant]
                    del mutant.fitness.values
            
            # 3. 评估无效的个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = parallel_evaluate(invalid_ind, pool)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 4. 更新种群
            pop[:] = offspring
            hof.update(pop)
            
            # 5. 记录统计数据
            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            print(logbook.stream)

    best_individual = hof[0]
    best_rm_values = sorted([round(x / CONFIG['rm_precision']) * CONFIG['rm_precision'] for x in best_individual if 0 < x < CONFIG['r_max']])
    
    # 重新计算最佳个体，获取 FWHM 和 SR
    best_res = evaluate_rm_sequence(best_rm_values)
    best_rm_values = sorted(np.unique(best_rm_values)).tolist()
    
    # 为了得到更多详细信息，需要再次运行 calculate_all_params (非并行)
    final_calculator = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])
    final_res = final_calculator.calculate_all_params(best_rm_values)

    print("\n=== 优化结果 ===")
    if final_res['success']:
        print(f"最佳 FWHM: {final_res['FWHM']:.4f} mm")
        print(f"最佳旁瓣比 (SR): {final_res['sidelobe_ratio']:.4f}")
        print(f"焦距: {final_res['focal_length']:.2f} mm")
        print(f"压缩比: {final_res['compression_ratio']:.4f} (目标 < 0.5)")
        print(f"优化后的 rm 序列 ({len(best_rm_values)} 个): \n{best_rm_values}")
    else:
        print("最佳个体计算失败。")

    print(f"\n总耗时: {time.time()-start_time:.1f}s")
    
    # 绘制适应度变化曲线
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, min_fitness, label='Minimum Fitness (FWHM + Penalty)', color='red')
    plt.plot(gen, avg_fitness, label='Average Fitness', color='blue')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (to be minimized)')
    plt.legend()
    plt.grid(True)
    figure_path = os.path.join(output_dir, 'ga_fitness_progress.png')
    plt.savefig(figure_path)
    print(f"GA 适应度图表已保存: {figure_path}")


if __name__ == "__main__":
    mp.freeze_support() # Windows/macOS 兼容
    
    parser = argparse.ArgumentParser(description="运行遗传算法优化")
    parser.add_argument("--pop-size", type=int, default=200, help="种群规模")
    parser.add_argument("--generations", type=int, default=50, help="迭代代数")
    parser.add_argument("--output-dir", default="outputs", help="结果输出目录")
    args = parser.parse_args()
    run_ga_optimization(pop_size=args.pop_size, num_generations=args.generations, output_dir=args.output_dir)
