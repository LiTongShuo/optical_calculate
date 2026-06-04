import numpy as np
from scipy.special import j0
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

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
        
        # 随机生成 r1~r20 {rm}，精度限制到0.01mm，允许大小相等
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 生成精度为0.01mm的随机点，允许重复
        self.r_points = np.round(np.random.uniform(0, self.r_max, 20), 2)
        self.r_points.sort()
        
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
    
    def find_focal_length(self, z_range=(10, 100), num_z_points=200):
        """
        寻找焦距：在r'=0的条件下，计算z在指定区间中I(r')的值，找到最大值对应的z值作为焦距f
        """
        z_values = np.linspace(z_range[0], z_range[1], num_z_points)
        intensities_at_origin = []
        
        print(f"正在计算焦距，扫描z范围: {z_range[0]} - {z_range[1]} mm")
        
        for i, z in enumerate(z_values):
            # 更新z值
            self.z = z
            self.k = 2 * np.pi / self.wavelength  # 更新波数
            
            # 计算r'=0处的强度
            r_prime_single = np.array([0.0])
            integrand_matrix = self.compute_integrand_matrix(r_prime_single)
            integral_result = simpson(integrand_matrix, self.r_integration, axis=1)[0]
            intensity = np.abs(integral_result)**2
            
            intensities_at_origin.append(intensity)
            
            if (i + 1) % 50 == 0:
                print(f"  已完成 {i + 1}/{num_z_points} 个z值")
        
        intensities_at_origin = np.array(intensities_at_origin)
        focal_idx = np.argmax(intensities_at_origin)
        focal_length = z_values[focal_idx]
        max_intensity = intensities_at_origin[focal_idx]
        
        print(f"找到焦距: f = {focal_length:.2f} mm, 最大强度: {max_intensity:.2e}")
        
        return focal_length, max_intensity, z_values, intensities_at_origin
    
    def calculate_compression_ratio(self, focal_length, r_prime_range=(-5, 5), num_r_prime_points=1000):
        """
        计算横向焦斑压缩比
        """
        # 设置到焦距位置
        self.z = focal_length
        
        # 计算强度分布
        r_prime_values, intensity_values = self.compute_intensity_profile_vectorized(
            r_prime_range, num_r_prime_points
        )
        
        # 找到最大强度Im和对应的r'位置
        max_intensity_idx = np.argmax(intensity_values)
        Im = intensity_values[max_intensity_idx]
        r_prime_center = r_prime_values[max_intensity_idx]
        
        print(f"最大强度 Im = {Im:.2e} 在 r' = {r_prime_center:.3f} mm")
        
        # 寻找半高全宽(FWHM)
        half_max = Im / 2
        
        # 在中心点右侧寻找半高位置
        right_side = intensity_values[max_intensity_idx:]
        right_r_prime = r_prime_values[max_intensity_idx:]
        right_half_idx = np.argmin(np.abs(right_side - half_max))
        r_right = right_r_prime[right_half_idx]
        
        # 在中心点左侧寻找半高位置
        left_side = intensity_values[:max_intensity_idx+1]
        left_r_prime = r_prime_values[:max_intensity_idx+1]
        left_half_idx = np.argmin(np.abs(left_side - half_max))
        r_left = left_r_prime[left_half_idx]
        
        Rm = abs(r_right - r_left)
        compression_ratio = Rm / self.wavelength
        
        print(f"半高全宽 Rm = {Rm:.3f} mm, 压缩比 = {compression_ratio:.3f}")
        
        return compression_ratio, Rm, Im, r_prime_values, intensity_values
    
    def calculate_sidelobe_ratio(self, focal_length, r_prime_range=(-5, 5), num_r_prime_points=1000):
        """
        计算横向旁瓣比
        """
        # 设置到焦距位置
        self.z = focal_length
        
        # 计算强度分布
        r_prime_values, intensity_values = self.compute_intensity_profile_vectorized(
            r_prime_range, num_r_prime_points
        )
        
        # 找到最大强度Im
        max_intensity_idx = np.argmax(intensity_values)
        Im = intensity_values[max_intensity_idx]
        
        # 使用find_peaks找到所有峰值
        peaks, properties = find_peaks(intensity_values, height=Im*0.01)  # 设置最小峰值高度
        
        # 排除主峰（r'=0附近的峰值）
        main_peak_region = 0.1  # 主峰区域范围(mm)
        sidelobe_peaks = peaks[np.abs(r_prime_values[peaks]) > main_peak_region]
        
        if len(sidelobe_peaks) > 0:
            # 找到旁瓣中的最大强度
            sidelobe_intensities = intensity_values[sidelobe_peaks]
            max_sidelobe_idx = np.argmax(sidelobe_intensities)
            I0 = sidelobe_intensities[max_sidelobe_idx]
            sidelobe_position = r_prime_values[sidelobe_peaks[max_sidelobe_idx]]
            
            sidelobe_ratio = I0 / Im
            
            print(f"最大旁瓣强度 I0 = {I0:.2e} 在 r' = {sidelobe_position:.3f} mm")
            print(f"旁瓣比 = {sidelobe_ratio:.4f}")
        else:
            I0 = 0
            sidelobe_ratio = 0
            sidelobe_position = 0
            print("未找到明显的旁瓣")
        
        return sidelobe_ratio, I0, Im, r_prime_values, intensity_values

def analyze_multiple_rm_sets(num_sets=10):
    """
    分析多组随机{rm}对应的光学参数
    """
    wavelength = 1.08  # mm
    results = []
    
    print("=" * 70)
    print(f"开始分析 {num_sets} 组随机 {{rm}} 的光学参数")
    print("=" * 70)
    
    for set_idx in range(num_sets):
        print(f"\n--- 第 {set_idx + 1} 组随机 {{rm}} ---")
        
        # 创建衍射计算器（使用不同的随机种子）
        random_seed = 1000 + set_idx * 100
        diffraction = OptimizedDiffractionIntegral(
            wavelength=wavelength,
            z=25,  # 初始z值，会被覆盖
            r_max=25,
            num_integration_points=1000,  # 减少点数以提高速度
            random_seed=random_seed
        )
        
        # 输出当前组的 {rm} 值
        print(f"随机 {{rm}} 值 (mm):")
        rm_str = "[" + ", ".join([f"{r:.2f}" for r in diffraction.r_points]) + "]"
        print(rm_str)
        
        # 检查是否有重复值
        unique_values = np.unique(diffraction.r_points)
        if len(unique_values) < len(diffraction.r_points):
            print(f"注意: 存在 {len(diffraction.r_points) - len(unique_values)} 个重复值")
        
        # 计算点间距信息
        distances = np.diff(diffraction.r_points)
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        print(f"点间距: 最小={min_distance:.3f} mm, 平均={avg_distance:.3f} mm")
        
        # 1. 寻找焦距
        focal_length, max_intensity, z_values, z_intensities = diffraction.find_focal_length(
            z_range=(10, 100), num_z_points=100
        )
        
        # 2. 计算横向焦斑压缩比
        compression_ratio, Rm, Im, r_prime_comp, intensity_comp = diffraction.calculate_compression_ratio(
            focal_length, r_prime_range=(-5, 5), num_r_prime_points=500
        )
        
        # 3. 计算横向旁瓣比
        sidelobe_ratio, I0, Im_sidelobe, r_prime_sidelobe, intensity_sidelobe = diffraction.calculate_sidelobe_ratio(
            focal_length, r_prime_range=(-5, 5), num_r_prime_points=500
        )
        
        # 存储结果
        result = {
            'set_index': set_idx + 1,
            'rm_values': diffraction.r_points.copy(),
            'focal_length': focal_length,
            'max_intensity': max_intensity,
            'compression_ratio': compression_ratio,
            'Rm': Rm,
            'sidelobe_ratio': sidelobe_ratio,
            'I0': I0,
            'Im': Im
        }
        
        results.append(result)
    
    # 输出汇总结果
    print("\n" + "=" * 70)
    print("汇总结果:")
    print("=" * 70)
    print(f"{'组别':<6} {'焦距(mm)':<10} {'压缩比':<12} {'旁瓣比':<12} {'Rm(mm)':<10} {'Im':<12}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['set_index']:<6} {result['focal_length']:<10.2f} "
              f"{result['compression_ratio']:<12.4f} {result['sidelobe_ratio']:<12.4f} "
              f"{result['Rm']:<10.3f} {result['Im']:<12.2e}")
    
    # 计算统计信息
    focal_lengths = [r['focal_length'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    sidelobe_ratios = [r['sidelobe_ratio'] for r in results]
    
    print("\n统计信息:")
    print(f"焦距平均值: {np.mean(focal_lengths):.2f} ± {np.std(focal_lengths):.2f} mm")
    print(f"压缩比平均值: {np.mean(compression_ratios):.4f} ± {np.std(compression_ratios):.4f}")
    print(f"旁瓣比平均值: {np.mean(sidelobe_ratios):.4f} ± {np.std(sidelobe_ratios):.4f}")
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    
    # 分析10组随机{rm}
    results = analyze_multiple_rm_sets(num_sets=10)
    
    end_time = time.time()
    print(f"\n总计算时间: {end_time - start_time:.2f} 秒")