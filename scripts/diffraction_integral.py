import numpy as np
from scipy.special import j0
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']

class OptimizedDiffractionIntegral:
    def __init__(self, wavelength=1.08, z=1000, r_max=25, num_integration_points=1000, random_seed=None):
        """
        初始化参数
        
        参数:
        wavelength: 波长 (mm)
        z: 传播距离 (mm)
        r_max: 最大积分半径 (mm)
        num_integration_points: 积分采样点数
        random_seed: 随机种子，用于生成可重复的随机 {rm}
        """
        self.wavelength = wavelength  # mm
        self.k0 = 2 * np.pi / wavelength  # 波数 k0
        self.k = self.k0  # 通常 k = k0
        self.z = z  # mm
        self.r_max = r_max
        self.num_integration_points = num_integration_points
        
        # 预计算积分网格
        self.r_integration = np.linspace(0, self.r_max, num_integration_points)
        
        # 随机生成 r1~r20 {rm}，从小到大排列
        if random_seed is not None:
            np.random.seed(random_seed)
        self.r_points = np.sort(np.random.uniform(0, self.r_max, 20))
        
        # 预计算 T(r) 在积分网格上的值
        self.T_values = self._precompute_T()
    
    def _precompute_T(self):
        """预计算 T(r) 在积分网格上的值"""
        T_arr = np.ones_like(self.r_integration)
        
        # 在 r1~r2, r3~r4, ..., r19~r20 区间内设为 -1
        for i in range(0, 19, 2):
            if i + 1 < len(self.r_points):
                mask = (self.r_integration >= self.r_points[i]) & (self.r_integration <= self.r_points[i+1])
                T_arr[mask] = -1
        
        return T_arr
    
    def T(self, r):
        """
        定义函数 T(r)，在特定区间内为-1，其余区间为1
        支持标量和数组输入
        """
        r_array = np.array(r)
        result = np.ones_like(r_array)
        
        # 在 r1~r2, r3~r4, ..., r19~r20 区间内设为 -1
        for i in range(0, 19, 2):
            if i + 1 < len(self.r_points):
                mask = (r_array >= self.r_points[i]) & (r_array <= self.r_points[i+1])
                result[mask] = -1
        
        return result
    
    def compute_integrand_matrix(self, r_prime_values):
        """
        向量化计算被积函数矩阵
        """
        num_r_prime = len(r_prime_values)
        integrand_matrix = np.zeros((num_r_prime, self.num_integration_points), dtype=complex)
        
        # 预计算一些公共项
        r_squared = self.r_integration**2
        T_r = self.T_values * self.r_integration  # T(r) * r
        
        for j, r_prime in enumerate(r_prime_values):
            r_prime_sq = r_prime**2
            
            # 计算平方根项 - 向量化
            sqrt_term = np.sqrt(r_prime_sq + r_squared + self.z**2)
            
            # 相位项 - 向量化
            phase = np.exp(1j * self.k * sqrt_term)
            
            # 贝塞尔函数项 - 向量化
            bessel_arg = (self.k * r_prime * self.r_integration) / sqrt_term
            bessel_term = j0(bessel_arg)
            
            # 完整的被积函数 - 向量化
            integrand_matrix[j, :] = (self.k0 / (1j * self.z)) * T_r * phase * bessel_term
        
        return integrand_matrix
    
    def compute_intensity_profile_vectorized(self, r_prime_range=None, num_r_prime_points=200):
        """
        向量化计算强度分布
        """
        if r_prime_range is None:
            r_prime_range = [-10 * self.wavelength, 10 * self.wavelength]  # 默认范围 [-10λ, 10λ]
        
        r_prime_values = np.linspace(r_prime_range[0], r_prime_range[1], num_r_prime_points)
        
        # 一次性计算所有被积函数
        integrand_matrix = self.compute_integrand_matrix(r_prime_values)
        
        # 向量化积分
        integral_results = simpson(integrand_matrix, self.r_integration, axis=1)
        
        # 计算强度
        intensity_values = np.abs(integral_results)**2
        
        return r_prime_values, intensity_values

def generate_multiple_rm_sets_and_plot():
    """生成几组随机 {rm} 并在5个不同z值下绘制I(r')的变化图表"""
    
    # 设置参数
    wavelength = 1.08  # mm
    z_values = [23, 24, 25, 26, 27]  # mm
    r_prime_range = [-10 * wavelength, 10 * wavelength]  # [-10λ, 10λ]
    num_r_prime_points = 500
    
    # 生成几组不同的随机 {rm}
    num_rm_sets = 3  # 生成3组不同的随机 {rm}
    
    # 创建图形
    fig, axes = plt.subplots(num_rm_sets, 1, figsize=(12, 4 * num_rm_sets))
    if num_rm_sets == 1:
        axes = [axes]
    
    for set_idx in range(num_rm_sets):
        print(f"计算第 {set_idx + 1} 组随机 {{rm}}...")
        
        # 为每组生成不同的随机种子
        random_seed = 42 + set_idx * 100
        
        # 存储所有z值的强度分布
        all_intensities = []
        
        for z in z_values:
            print(f"  计算 z = {z} mm...")
            
            # 创建衍射计算器
            diffraction = OptimizedDiffractionIntegral(
                wavelength=wavelength,
                z=z,
                r_max=25,
                num_integration_points=2000,
                random_seed=random_seed
            )
            
            # 计算强度分布
            r_prime_values, intensity_values = diffraction.compute_intensity_profile_vectorized(
                r_prime_range, 
                num_r_prime_points
            )
            
            all_intensities.append(intensity_values)
        
        # 绘制当前 {rm} 组的所有z值结果
        ax = axes[set_idx]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (z, intensity) in enumerate(zip(z_values, all_intensities)):
            ax.plot(r_prime_values / wavelength, intensity, 
                   color=colors[i], linewidth=2, label=f'z = {z}mm')
        
        ax.set_xlabel("r' / λ")
        ax.set_ylabel("I(r')")
        ax.set_title(f"随机 {{rm}} 组 {set_idx + 1} 在不同z值下的强度分布")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 输出当前组的 {rm} 值
        diffraction_temp = OptimizedDiffractionIntegral(random_seed=random_seed)
        print(f"第 {set_idx + 1} 组 {{rm}} 值:")
        print("[" + ", ".join([f"{r:.2f}" for r in diffraction_temp.r_points]) + "]")
        print()
    
    plt.tight_layout()
    plt.show()

def generate_single_rm_set_detailed_analysis():
    """生成一组随机 {rm} 并进行详细分析"""
    
    wavelength = 1.08  # mm
    z_values = [23, 24, 25, 26, 27]  # mm
    random_seed = 123  # 固定随机种子
    
    # 创建衍射计算器获取 {rm} 值
    diffraction = OptimizedDiffractionIntegral(random_seed=random_seed)
    
    print("随机生成的 {rm} 值 (mm):")
    print("[" + ", ".join([f"{r:.4f}" for r in diffraction.r_points]) + "]")
    print()
    
    # 计算所有z值的强度分布
    all_results = []
    
    for z in z_values:
        print(f"计算 z = {z} mm 的强度分布...")
        
        diffraction_z = OptimizedDiffractionIntegral(
            wavelength=wavelength,
            z=z,
            r_max=25,
            num_integration_points=2000,
            random_seed=random_seed
        )
        
        r_prime_range = [-10 * wavelength, 10 * wavelength]
        r_prime_values, intensity_values = diffraction_z.compute_intensity_profile_vectorized(
            r_prime_range, 
            num_r_prime_points=500
        )
        
        all_results.append((r_prime_values, intensity_values, z))
    
    # 绘制结果
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (r_prime_vals, intensity_vals, z) in enumerate(all_results):
        plt.semilogy(r_prime_vals / wavelength, intensity_vals, 
                    color=colors[i], linewidth=2, label=f'z = {z}mm')
    
    plt.xlabel("r' / λ")
    plt.ylabel("I(r')")
    plt.title("随机 {rm} 在不同z值下的衍射强度分布")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # 输出统计信息
    print("\n强度分布统计:")
    for i, (r_prime_vals, intensity_vals, z) in enumerate(all_results):
        max_idx = np.argmax(intensity_vals)
        max_intensity = intensity_vals[max_idx]
        max_r_prime = r_prime_vals[max_idx] / wavelength
        print(f"z = {z}mm: 最大强度 {max_intensity:.2e} 在 r' = {max_r_prime:.2f}λ")

if __name__ == "__main__":
    print("开始计算多组随机 {rm} 在不同z值下的强度分布...")
    print("=" * 60)
    
    # 生成多组随机 {rm} 并绘图
    generate_multiple_rm_sets_and_plot()
    
    print("=" * 60)
    print("完成!")