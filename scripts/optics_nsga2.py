# optics_nsga2.py

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.integrate import simpson
from scipy.special import j0
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import time
import warnings
import argparse
import os
# 引入 simulation.py 中的所有必要模块
import matplotlib.pyplot as plt # NSGA-II 最终绘图需要，但并行评估中不使用
import seaborn as sns
# from OpticalParamsCalculator import OpticalParamsCalculator, CONFIG # 假设这两个已定义

# 忽略警告
warnings.filterwarnings('ignore')

CONFIG = {
    'wavelength': 1.08,           # 波长 (mm)
    'r_max': 25.0,                # 最大半径 (mm)
    'rm_count': 20,               # 固定 rm 数量
    'rm_precision': 0.01,         # 加工精度/模拟精度
    'num_integration_points': 1000, # 积分精度 (提高以保证物理准确性)
    
    # 焦距搜索设置
    'z_range': (5, 50),         
    'num_z_points': 150,          
    
    # 横向场设置 (关键改进点)
    'r_prime_range': (-2.0, 2.0), # 缩小范围专注中心，减少计算量
    'num_r_prime_points': 5000,   # 采样点数 5000 -> 步长 0.0012mm (极高精度)
}

# ================= 核心物理计算类 =================
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


# 辅助函数：计算适应度 (NSGA-II 版本)
def evaluate_rm_sequence_nsga2(rm_values: list):
    """
    改进的评估函数：
    1. 增加 0.01mm 加工精度约束
    2. 剔除卡在焦距搜索边界的解
    """
    # --- 1. 加工精度约束 ---
    precision = CONFIG.get('rm_precision', 0.01)
    rm_values = [round(x / precision) * precision for x in rm_values]
    rm_values = np.sort(np.unique(rm_values)).tolist()
    
    # 保持偶数序列
    if len(rm_values) % 2 != 0:
        rm_values = rm_values[:-1]
    
    if len(rm_values) < 2:
        return (2.16, 1.0) 

    try:
        calculator = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])
        res = calculator.calculate_all_params(rm_values)

        if not res['success']:
            return (2.16, 1.0)
        
        # --- 2. 焦距边界剔除逻辑 ---
        f_len = res['focal_length']
        z_min, z_max = CONFIG['z_range']
        margin = 2.0  # 设置 2mm 的缓冲区
        
        # 如果焦距距离边界太近，则赋予惩罚值，让其无法进入 Pareto 前沿
        if f_len <= (z_min + margin) or f_len >= (z_max - margin):
            return (2.16, 1.0) 
        
        return (res['FWHM'], res['sidelobe_ratio'])
    except Exception:
        return (2.16, 1.0)

# 辅助函数：并行评估
def parallel_evaluate(individuals, pool):
    """
    使用多进程并行评估 DEAP 中的所有个体。
    """
    # 提取个体基因
    tasks = [list(ind) for ind in individuals]
    # 使用 imap_unordered 以便在 tqdm 中显示进度
    results = list(tqdm(pool.imap_unordered(evaluate_rm_sequence_nsga2, tasks), 
                        total=len(individuals), desc="评估代数"))
    return results

# ================= 遗传算法配置 (NSGA-II) =================

# 1. 设置优化目标: 两个最小化目标 ('weights': (-1.0, -1.0))
# 第一个目标：FWHM (最小化)
# 第二个目标：Sidelobe Ratio (SR) (最小化)
creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0, -1.0))
# 2. 个体 (Individual): 列表形式，带上 Fitness 属性
creator.create("Individual", list, fitness=creator.FitnessMinMulti)

# 初始化工具箱
toolbox = base.Toolbox()

# 基因生成器: 随机浮点数 (0 到 r_max 之间)
toolbox.register("attr_float", np.random.uniform, 0.0, CONFIG['r_max'])

# 个体初始化
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=CONFIG['rm_count'])

# 种群初始化
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# GA 算子注册
toolbox.register("evaluate", evaluate_rm_sequence_nsga2) 

# 选择: NSGA-II 的选择算子
toolbox.register("select", tools.selNSGA2) 

# 交叉: 两点交叉
toolbox.register("mate", tools.cxTwoPoint)

# 变异: 高斯变异
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2.0, indpb=0.1)

# ================= 主程序 (NSGA-II 运行) =================
def run_nsga2_optimization(pop_size=100, num_generations=50, cxpb=0.7, mutpb=0.2, output_dir="outputs"):
    
    os.makedirs(output_dir, exist_ok=True)

    # 在 run_nsga2_optimization 内部定义
    def mutate_with_precision(individual):
        # 先执行标准的高斯变异
        tools.mutGaussian(individual, mu=0, sigma=2.0, indpb=0.1)
        # 变异后立即修整精度
        for i in range(len(individual)):
            individual[i] = round(individual[i] / CONFIG['rm_precision']) * CONFIG['rm_precision']
        return (individual,)
    
    # 注册新的变异算子
    toolbox.register("mutate", mutate_with_precision)

    start_time = time.time()
    print("=== 超振荡透镜 NSGA-II 多目标优化开始 ===")
    print(f"种群大小: {pop_size} | 迭代次数: {num_generations} | 基因数: {CONFIG['rm_count']}")
    
    pool_size = max(1, mp.cpu_count() - 1) 
    print(f"正在计算... (使用 {pool_size} 核心进行评估)")
    
    # 创建初始种群
    pop = toolbox.population(n=pop_size)
    
    expert_rm = [1.94, 2.44, 3.97, 4.06, 4.09, 4.62, 5.06, 15.5, 16.62, 18.64, 19.03, 19.14, 19.37, 21.29, 22.07, 22.64, 22.82, 23.5, 24.28, 24.67]
    
    # 将第一个个体替换为你的手动序列
    pop[0] = creator.Individual(expert_rm)

    
    # 记录统计信息 (记录 FWHM 和 SR 的最小值)
    # 我们不能使用 sel 关键字，而是将选择操作集成到 lambda 函数中
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    # 最小 FWHM (目标 0)
    stats.register("min_FWHM", lambda x: np.min([i[0] for i in x]))
    # 最小 SR (目标 1)
    stats.register("min_SR", lambda x: np.min([i[1] for i in x]))
    # 平均 FWHM (目标 0)
    stats.register("avg_FWHM", lambda x: np.mean([i[0] for i in x]))

    logbook = tools.Logbook()
    
    # Hall of Fame 记录非支配解集 (即 Pareto 前沿)
    hof = tools.ParetoFront()

    # 多进程评估
    with mp.Pool(processes=pool_size) as pool:
        
        # 第一次评估
        fitnesses = parallel_evaluate(pop, pool)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # 使用 NSGA-II 流程进行优化
        # eaMuPlusLambda: 综合了 μ 个父代和 λ 个子代来形成下一代 (μ+λ)
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, 
                                                 mu=pop_size, # 父代数量
                                                 lambda_=pop_size, # 子代数量
                                                 cxpb=cxpb, 
                                                 mutpb=mutpb, 
                                                 ngen=num_generations, 
                                                 stats=stats, 
                                                 halloffame=hof, 
                                                 verbose=True)

    print("\n=== 优化结果 (Pareto 前沿) ===")
    
    # 提取 Pareto 前沿上的解
    pareto_solutions = []
    final_calculator = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])

    for individual in hof:
        # 重新格式化 rm 序列 (去重、排序、精度修正)
        rm_values = sorted([round(x / CONFIG['rm_precision']) * CONFIG['rm_precision'] 
                            for x in individual if 0 < x < CONFIG['r_max']])
        rm_values = sorted(np.unique(rm_values).tolist())
        
        # 重新计算以获得完整的物理参数
        try:
            res = final_calculator.calculate_all_params(rm_values)
            if res['success']:
                pareto_solutions.append({
                    "FWHM (mm)": res['FWHM'],
                    "SR": res['sidelobe_ratio'],
                    "Focal Length (mm)": res['focal_length'],
                    "rm Sequence": rm_values
                })
        except Exception:
            # 忽略计算失败的个体
            pass

    # 打印非支配解 (Pareto Front)
    if pareto_solutions:
        # 将结果整理成 Pandas DataFrame
        results_df = pd.DataFrame(pareto_solutions)
        results_df.sort_values(by=['FWHM (mm)', 'SR'], inplace=True)
        
        print(f"找到 {len(results_df)} 个非支配解 (FWHM vs SR)：\n")
        
        # 打印关键指标，省略 rm 序列
        print(results_df[['FWHM (mm)', 'SR', 'Focal Length (mm)']].to_markdown(index=False, numalign="left", stralign="left"))
        
        # 找出最具超分辨能力的解 (FWHM最小)
        best_fwhm_solution = results_df.iloc[0]
        print("\n--- 最佳 FWHM 解 ---")
        print(f"FWHM: {best_fwhm_solution['FWHM (mm)']:.4f} mm")
        print(f"旁瓣比 (SR): {best_fwhm_solution['SR']:.4f}")
        print(f"焦距: {best_fwhm_solution['Focal Length (mm)']:.2f} mm")
        print(f"优化后的 rm 序列 ({len(best_fwhm_solution['rm Sequence'])} 个): \n{best_fwhm_solution['rm Sequence']}")
        
        # 绘制 Pareto 前沿图
        if 'matplotlib.pyplot' in globals() and 'seaborn' in globals():
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='FWHM (mm)', y='SR', data=results_df, 
                            hue='Focal Length (mm)', palette='viridis', s=100)
            plt.title('Pareto Front: FWHM vs Sidelobe Ratio (NSGA-II)', fontsize=14)
            plt.xlabel('FWHM (mm)', fontsize=12)
            plt.ylabel('Sidelobe Ratio (SR)', fontsize=12)
            plt.axvline(x=0.5 * CONFIG['wavelength'], color='r', linestyle='--', label='0.5λ Limit')
            plt.legend()
            plt.grid(True)
            figure_path = os.path.join(output_dir, 'nsga2_pareto_front.png')
            plt.savefig(figure_path)
            print(f"\nPareto 前沿图表已保存: {figure_path}")
        
    else:
        print("未找到有效的非支配解。")


    print(f"\n总耗时: {time.time()-start_time:.1f}s")

def plot_2d_propagation_map(best_rm, output_dir="outputs", show=False):
    """
    绘制 R-Z 平面的 2D 强度分布图
    """
    os.makedirs(output_dir, exist_ok=True)
    calc = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])
    T_values = calc.compute_T_values(best_rm)
    
    # 获取当前序列的焦距
    res_base = calc.calculate_all_params(best_rm)
    f_z = res_base['focal_length']
    
    # 定义绘图区域：焦点前后各 5mm，横向中心 1.5mm
    z_start = max(0.1, f_z - 5.0)
    z_end = f_z + 5.0
    z_axis = np.linspace(z_start, z_end, 100)
    r_axis = np.linspace(-1.5, 1.5, 200)
    
    intensity_map = np.zeros((len(r_axis), len(z_axis)))
    
    print(f"正在生成 2D 传播图数据 (焦点 Z={f_z:.2f}mm)...")
    for j, z in enumerate(tqdm(z_axis)):
        for i, r in enumerate(r_axis):
            integrand = calc.compute_integrand(r, z, T_values)
            intensity_map[i, j] = np.abs(simpson(integrand, calc.r_integration))**2
            
    # 归一化并绘图
    intensity_map /= np.max(intensity_map)
    
    plt.figure(figsize=(12, 6))
    # 使用 'inferno' 颜色图，能更好地表现对比度
    pcm = plt.pcolormesh(z_axis, r_axis, intensity_map, cmap='inferno', shading='gouraud')
    plt.colorbar(pcm, label='Normalized Intensity')
    
    plt.title(f'2D Field Propagation (Optical Needle)\nFWHM={res_base["FWHM"]:.4f}mm, Focal Z={f_z:.2f}mm', fontsize=14)
    plt.xlabel('Propagation Distance Z (mm)', fontsize=12)
    plt.ylabel('Transverse Radius R (mm)', fontsize=12)
    
    # 标记焦点位置
    plt.axvline(x=f_z, color='white', linestyle='--', alpha=0.5)
    plt.tight_layout()
    figure_path = os.path.join(output_dir, 'nsga2_propagation_map.png')
    plt.savefig(figure_path, dpi=300)
    print(f"2D 传播图已保存: {figure_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    mp.freeze_support() # Windows/macOS 兼容
    parser = argparse.ArgumentParser(description="运行 NSGA-II 优化或绘制传播图")
    parser.add_argument("--mode", choices=["plot", "optimize"], default="plot", help="运行模式")
    parser.add_argument("--pop-size", type=int, default=100, help="种群规模")
    parser.add_argument("--generations", type=int, default=20, help="迭代代数")
    parser.add_argument("--output-dir", default="outputs", help="结果输出目录")
    parser.add_argument("--show", action="store_true", help="保存图片后同时显示窗口")
    args = parser.parse_args()
    
    # 运行 NSGA-II
    # 使用更大的种群和迭代次数以获得更精细的 Pareto 前沿
    if args.mode == "optimize":
        run_nsga2_optimization(pop_size=args.pop_size, num_generations=args.generations, output_dir=args.output_dir)
    else:
        best_rm_found = [0.04, 1.81, 2.34, 4.38, 7.35, 8.36, 9.07, 12.46, 13.15, 13.53, 13.97, 14.96, 15.41, 16.35, 19.23, 19.55, 22.09, 22.25, 23.88, 24.47]
        #best_rm_found = [1.03, 1.54, 2.0, 2.89, 3.25, 5.53, 8.8, 9.19, 10.950000000000001, 12.46, 13.22, 13.71, 13.780000000000001, 14.98, 17.35, 17.52, 20.73, 23.43, 23.96, 24.64]
        #best_rm_found = [1.94, 2.44, 3.97, 4.0600000000000005, 4.09, 4.62, 5.0600000000000005, 15.5, 16.62, 18.64, 19.03, 19.14, 19.37, 21.29, 22.07, 22.64, 22.82, 23.5, 24.28, 24.67]
        plot_2d_propagation_map(best_rm_found, output_dir=args.output_dir, show=args.show)
    
