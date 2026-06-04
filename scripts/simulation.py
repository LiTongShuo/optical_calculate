import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import j0
from scipy.integrate import simpson
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
import time
import argparse
import os

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
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

# ================= 辅助函数 =================
def generate_fixed_rm_sequence() -> list:
    rm_values = np.random.uniform(0, CONFIG['r_max'], CONFIG['rm_count'])
    rm_values = np.round(rm_values / CONFIG['rm_precision']) * CONFIG['rm_precision']
    rm_values = np.sort(rm_values)
    return rm_values.tolist()

def worker_process(args):
    idx, rm_values = args
    calculator = OpticalParamsCalculator(CONFIG['wavelength'], CONFIG['r_max'], CONFIG['num_integration_points'])
    res = calculator.calculate_all_params(rm_values)
    return {**res, 'rm_values': rm_values}

# ================= 主程序 =================
def main(num_samples=2000, output_dir="outputs"):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    print(f"=== 高精度超透镜模拟 (Interp + High Res) ===")
    print(f"目标样本: {num_samples} | 横向采样: {CONFIG['num_r_prime_points']} 点")

    # 1. 生成任务
    tasks = [(i, generate_fixed_rm_sequence()) for i in range(num_samples)]

    # 2. 并行计算
    results = []
    # 稍微减少进程数以避免内存溢出，因为现在单次计算量大了
    pool_size = max(1, mp.cpu_count() - 2) 
    print(f"正在计算... (使用 {pool_size} 核心)")
    
    with mp.Pool(processes=pool_size) as pool:
        for res in tqdm(pool.imap_unordered(worker_process, tasks, chunksize=5), total=num_samples):
            results.append(res)
    
    df = pd.DataFrame(results)
    df = df[df['success'] == True]

    # 3. 筛选数据
    fwhm_limit = 0.5 * CONFIG['wavelength']
    filtered_df = df[
        (df['FWHM'] <= fwhm_limit) & 
        (df['focal_length'] > 10) & 
        (df['focal_length'] < 50)
    ].copy()
    
    print(f"\n原始数据: {len(df)} -> 筛选后: {len(filtered_df)}")
    
    if filtered_df.empty:
        print("没有满足条件的数据。")
        return

    # 保存数据
    results_path = os.path.join(output_dir, 'high_precision_results.csv')
    filtered_df.to_csv(results_path, index=False)

    # 4. 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 直方图: 增加 bins 数量，观察是否平滑
    sns.histplot(data=filtered_df, x='FWHM', bins=50, kde=True, 
                 ax=ax1, color='dodgerblue', edgecolor='k', alpha=0.6)
    ax1.axvline(x=fwhm_limit, color='r', linestyle='--', linewidth=2, label='0.5$\\lambda$ Limit')
    ax1.set_title('FWHM Distribution (High Precision)', fontsize=14)
    ax1.set_xlabel('FWHM (mm)', fontsize=12)
    ax1.legend()

    # 散点图
    sc = ax2.scatter(filtered_df['FWHM'], filtered_df['sidelobe_ratio'], 
                     c=filtered_df['focal_length'], cmap='viridis', s=20, alpha=0.6)
    ax2.set_title('Sidelobe Ratio vs FWHM', fontsize=14)
    ax2.set_xlabel('FWHM (mm)', fontsize=12)
    ax2.set_ylabel('Sidelobe Ratio (Intensity)', fontsize=12)
    
    # 辅助线 (0.4波长压缩比)
    target_fwhm = 0.4 * CONFIG['wavelength']
    ax2.axvline(x=target_fwhm, color='orange', linestyle='--', label=f'0.4$\\lambda$ ({target_fwhm:.3f}mm)')
    ax2.axhline(y=0.2, color='gray', linestyle=':', label='SR = 0.2')
    
    plt.colorbar(sc, ax=ax2, label='Focal Length (mm)')
    ax2.legend()
    
    plt.tight_layout()
    figure_path = os.path.join(output_dir, 'high_precision_analysis.png')
    plt.savefig(figure_path, dpi=300)
    print(f"结果已保存: {results_path}")
    print(f"图表已保存: {figure_path}")
    print(f"耗时: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="运行高精度超透镜随机仿真")
    parser.add_argument("--num-samples", type=int, default=3000, help="随机样本数")
    parser.add_argument("--output-dir", default="outputs", help="结果输出目录")
    args = parser.parse_args()
    main(num_samples=args.num_samples, output_dir=args.output_dir)
