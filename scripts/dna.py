# ----------------------------------------------------------------
# 解决方案第一步：定义物镜函数
# ----------------------------------------------------------------
from scipy.optimize import differential_evolution
import numpy as np
import pandas as pd
import json
import os
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional
import traceback
from tqdm import tqdm
import time
from scipy.special import j0
from scipy.integrate import simpson
from scipy.signal import find_peaks
import warnings

# 假设您的计算代码（OpticalParamsCalculator等）已经定义好
class OpticalParamsCalculator:
    """
    光学参数计算器 - 基于衍射积分计算焦距、压缩比和旁瓣比
    """
    
    def __init__(self, wavelength: float = 1.08, r_max: float = 25.0, 
                 num_integration_points: int = 800):
        """
        初始化光学参数计算器
        
        参数:
        wavelength: 波长 (mm)
        r_max: 最大积分半径 (mm)
        num_integration_points: 积分采样点数
        """
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength
        self.r_max = r_max
        self.num_integration_points = num_integration_points
        
        # 预计算积分网格
        self.r_integration = np.linspace(0, self.r_max, num_integration_points)
    
    def compute_T_values(self, rm_values: List[float]) -> np.ndarray:
        """
        计算T(r)在积分网格上的值
        """
        T_arr = np.ones_like(self.r_integration)
        
        # 在 r1~r2, r3~r4, ..., 区间内设为 -1
        for i in range(0, len(rm_values) - 1, 2):
            if i + 1 < len(rm_values):
                mask = (self.r_integration >= rm_values[i]) & (self.r_integration <= rm_values[i+1])
                T_arr[mask] = -1
        
        return T_arr
    
    def compute_integrand(self, r_prime: float, z: float, T_values: np.ndarray) -> np.ndarray:
        """
        计算单个r'值的被积函数
        """
        k = 2 * np.pi / self.wavelength
        
        # 预计算公共项
        r_squared = self.r_integration**2
        T_r = T_values * self.r_integration
        
        # 计算各项
        sqrt_term = np.sqrt(r_prime**2 + r_squared + z**2)
        phase = np.exp(1j * k * sqrt_term)
        bessel_arg = (k * r_prime * self.r_integration) / sqrt_term
        bessel_term = j0(bessel_arg)
        
        # 完整的被积函数
        integrand = (self.k0 / (1j * z)) * T_r * phase * bessel_term
        
        return integrand
    
    def find_focal_length(self, rm_values: List[float], 
                         z_range: Tuple[float, float] = (0, 100),
                         num_z_points: int = 80) -> Tuple[float, float]:
        """
        寻找焦距
        
        返回:
        focal_length: 焦距 (mm)
        max_intensity: 最大强度
        """
        try:
            # 预计算T_values
            T_values = self.compute_T_values(rm_values)
            
            z_values = np.linspace(z_range[0], z_range[1], num_z_points)
            intensities = []
            
            for z in z_values:
                # 计算r'=0处的强度
                integrand = self.compute_integrand(0.0, z, T_values)
                integral_result = simpson(integrand, self.r_integration)
                intensity = np.abs(integral_result)**2
                intensities.append(intensity)
            
            intensities = np.array(intensities)
            focal_idx = np.argmax(intensities)
            focal_length = z_values[focal_idx]
            max_intensity = intensities[focal_idx]
            
            return focal_length, max_intensity
            
        except Exception as e:
            raise RuntimeError(f"焦距计算失败: {str(e)}")
    
    def calculate_compression_ratio(self, rm_values: List[float], focal_length: float,
                                  r_prime_range: Tuple[float, float] = (-5, 5),
                                  num_r_prime_points: int = 400) -> Tuple[float, float, float]:
        """
        计算横向焦斑压缩比
        
        返回:
        compression_ratio: 压缩比
        Rm: 半高全宽 (mm)
        Im: 最大强度
        """
        try:
            # 预计算T_values
            T_values = self.compute_T_values(rm_values)
            
            r_prime_values = np.linspace(r_prime_range[0], r_prime_range[1], num_r_prime_points)
            intensities = []
            
            for r_prime in r_prime_values:
                integrand = self.compute_integrand(r_prime, focal_length, T_values)
                integral_result = simpson(integrand, self.r_integration)
                intensity = np.abs(integral_result)**2
                intensities.append(intensity)
            
            intensities = np.array(intensities)
            
            # 找到最大强度
            max_intensity_idx = np.argmax(intensities)
            Im = intensities[max_intensity_idx]
            
            # 寻找半高全宽(FWHM)
            half_max = Im / 2
            
            # 右侧寻找半高位置
            right_side = intensities[max_intensity_idx:]
            right_r_prime = r_prime_values[max_intensity_idx:]
            right_half_idx = np.argmin(np.abs(right_side - half_max))
            r_right = right_r_prime[right_half_idx]
            
            # 左侧寻找半高位置
            left_side = intensities[:max_intensity_idx+1]
            left_r_prime = r_prime_values[:max_intensity_idx+1]
            left_half_idx = np.argmin(np.abs(left_side - half_max))
            r_left = left_r_prime[left_half_idx]
            
            Rm = abs(r_right - r_left)
            compression_ratio = Rm / self.wavelength
            
            return compression_ratio, Rm, Im
            
        except Exception as e:
            raise RuntimeError(f"压缩比计算失败: {str(e)}")
    
    def calculate_sidelobe_ratio(self, rm_values: List[float], focal_length: float,
                               r_prime_range: Tuple[float, float] = (-5, 5),
                               num_r_prime_points: int = 400) -> Tuple[float, float, float]:
        """
        计算横向旁瓣比
        
        返回:
        sidelobe_ratio: 旁瓣比
        I0: 最大旁瓣强度
        Im: 主瓣强度
        """
        try:
            # 预计算T_values
            T_values = self.compute_T_values(rm_values)
            
            r_prime_values = np.linspace(r_prime_range[0], r_prime_range[1], num_r_prime_points)
            intensities = []
            
            for r_prime in r_prime_values:
                integrand = self.compute_integrand(r_prime, focal_length, T_values)
                integral_result = simpson(integrand, self.r_integration)
                intensity = np.abs(integral_result)**2
                intensities.append(intensity)
            
            intensities = np.array(intensities)
            
            # 找到主瓣强度
            max_intensity_idx = np.argmax(intensities)
            Im = intensities[max_intensity_idx]
            
            # 使用find_peaks找到所有峰值
            peaks, _ = find_peaks(intensities, height=Im * 0.01, distance=10)
            
            # 排除主峰（r'=0附近的峰值）
            main_peak_region = 0.2  # 主峰区域范围(mm)
            sidelobe_peaks = peaks[np.abs(r_prime_values[peaks]) > main_peak_region]
            
            if len(sidelobe_peaks) > 0:
                # 找到旁瓣中的最大强度
                sidelobe_intensities = intensities[sidelobe_peaks]
                max_sidelobe_idx = np.argmax(sidelobe_intensities)
                I0 = sidelobe_intensities[max_sidelobe_idx]
                sidelobe_ratio = I0 / Im
            else:
                I0 = 0.0
                sidelobe_ratio = 0.0
            
            return sidelobe_ratio, I0, Im
            
        except Exception as e:
            raise RuntimeError(f"旁瓣比计算失败: {str(e)}")

def process_single_sample(args: Tuple[Dict, Dict]) -> Dict[str, Any]:
    """
    处理单个样本的包装函数，用于并行计算
    
    参数:
    args: (sample, config) 元组
    
    返回:
    result: 包含计算结果和原始数据的字典
    """
    sample, config = args
    calculator_config = config['calculator']
    
    try:
        # 创建计算器实例
        calculator = OpticalParamsCalculator(
            wavelength=calculator_config['wavelength'],
            r_max=calculator_config['r_max'],
            num_integration_points=calculator_config['num_integration_points']
        )
        
        # 解析rm_values
        if isinstance(sample['rm_values'], str):
            rm_values = json.loads(sample['rm_values'])
        else:
            rm_values = sample['rm_values']
        
        # 数据验证
        if not isinstance(rm_values, list) or len(rm_values) < 2:
            raise ValueError(f"无效的rm_values: {rm_values}")
        
        if any(r < 0 or r > calculator_config['r_max'] for r in rm_values):
            raise ValueError(f"rm_values超出范围: {rm_values}")
        
        # 1. 寻找焦距
        focal_length, max_intensity_z = calculator.find_focal_length(
            rm_values, 
            z_range=config['z_range'],
            num_z_points=config['num_z_points']
        )
        
        # 2. 计算横向焦斑压缩比
        compression_ratio, Rm, Im = calculator.calculate_compression_ratio(
            rm_values, focal_length,
            r_prime_range=config['r_prime_range'],
            num_r_prime_points=config['num_r_prime_points']
        )
        
        # 3. 计算横向旁瓣比
        sidelobe_ratio, I0, Im_sidelobe = calculator.calculate_sidelobe_ratio(
            rm_values, focal_length,
            r_prime_range=config['r_prime_range'],
            num_r_prime_points=config['num_r_prime_points']
        )
        
        # 构建结果
        result = {
            'set_id': sample['set_id'],
            'rm_count': sample['rm_count'],
            'rm_values': rm_values,
            'rm_mean': sample.get('rm_mean', np.mean(rm_values)),
            'rm_std': sample.get('rm_std', np.std(rm_values)),
            'rm_min': sample.get('rm_min', np.min(rm_values)),
            'rm_max': sample.get('rm_max', np.max(rm_values)),
            'r1': sample.get('r1', rm_values[0]),
            'r1_is_zero': sample.get('r1_is_zero', False),
            
            # 计算结果
            'focal_length': float(focal_length),
            'max_intensity_z': float(max_intensity_z),
            'compression_ratio': float(compression_ratio),
            'Rm': float(Rm),
            'Im': float(Im),
            'sidelobe_ratio': float(sidelobe_ratio),
            'I0': float(I0),
            'Im_sidelobe': float(Im_sidelobe),
            
            # 状态标记
            'success': True,
            'error_message': None
        }
        
        return result
        
    except Exception as e:
        # 错误处理
        error_result = {
            'set_id': sample['set_id'],
            'rm_count': sample.get('rm_count', 0),
            'rm_values': sample.get('rm_values', []),
            'success': False,
            'error_message': f"{type(e).__name__}: {str(e)}",
            'focal_length': 0.0,
            'compression_ratio': 0.0,
            'sidelobe_ratio': 0.0,
            'Rm': 0.0,
            'Im': 0.0,
            'I0': 0.0
        }
        return error_result

class ParallelOpticalCalculator:
    """
    并行光学计算器
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化并行计算器
        """
        self.config = config or self.get_default_config()
        
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'calculator': {
                'wavelength': 1.08,
                'r_max': 25.0,
                'num_integration_points': 800
            },
            'z_range': (10, 100),
            'num_z_points': 80,
            'r_prime_range': (-5, 5),
            'num_r_prime_points': 400,
            'parallel': {
                'num_processes': max(1, mp.cpu_count() - 1),  # 留一个核心给系统
                'chunk_size': 10
            },
            'validation': {
                'max_rm_count': 50,
                'min_rm_count': 2,
                'max_r_value': 25.0
            }
        }
    
    def validate_dataset(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        验证数据集
        
        返回:
        valid_samples: 有效样本列表
        invalid_samples: 无效样本列表
        """
        valid_samples = []
        invalid_samples = []
        
        for sample in dataset:
            try:
                # 解析rm_values
                if isinstance(sample['rm_values'], str):
                    rm_values = json.loads(sample['rm_values'])
                else:
                    rm_values = sample['rm_values']
                
                # 验证基本字段
                if 'set_id' not in sample:
                    raise ValueError("缺少set_id字段")
                
                if not isinstance(rm_values, list):
                    raise ValueError("rm_values必须是列表")
                
                # 验证rm_values
                rm_count = len(rm_values)
                if rm_count < self.config['validation']['min_rm_count']:
                    raise ValueError(f"rm_count太小: {rm_count}")
                
                if rm_count > self.config['validation']['max_rm_count']:
                    raise ValueError(f"rm_count太大: {rm_count}")
                
                if any(r < 0 or r > self.config['validation']['max_r_value'] for r in rm_values):
                    raise ValueError("rm_values包含超出范围的值")
                
                if not all(isinstance(r, (int, float)) for r in rm_values):
                    raise ValueError("rm_values包含非数值")
                
                # 验证排序
                if rm_values != sorted(rm_values):
                    raise ValueError("rm_values未排序")
                
                valid_samples.append(sample)
                
            except Exception as e:
                sample['validation_error'] = str(e)
                invalid_samples.append(sample)
        
        return valid_samples, invalid_samples
    
    def compute_optical_params(self, dataset: List[Dict], 
                             show_progress: bool = True) -> List[Dict]:
        """
        并行计算光学参数
        
        参数:
        dataset: 输入数据集
        show_progress: 是否显示进度条
        
        返回:
        results: 计算结果列表
        """
        print("开始数据验证...")
        valid_samples, invalid_samples = self.validate_dataset(dataset)
        
        print(f"数据验证完成:")
        print(f"  有效样本: {len(valid_samples)}")
        print(f"  无效样本: {len(invalid_samples)}")
        
        if invalid_samples:
            print("\n无效样本详情:")
            for sample in invalid_samples[:5]:  # 只显示前5个
                print(f"  set_id={sample.get('set_id', 'N/A')}: {sample.get('validation_error', 'Unknown error')}")
            if len(invalid_samples) > 5:
                print(f"  ... 还有 {len(invalid_samples) - 5} 个无效样本")
        
        if not valid_samples:
            raise ValueError("没有有效样本可供计算")
        
        # 准备并行计算参数
        num_processes = self.config['parallel']['num_processes']
        chunk_size = self.config['parallel']['chunk_size']
        
        print(f"\n开始并行计算 (进程数: {num_processes})...")
        
        # 准备任务参数
        tasks = [(sample, self.config) for sample in valid_samples]
        
        results = []
        failed_count = 0
        
        # 使用进程池并行计算
        with mp.Pool(processes=num_processes) as pool:
            if show_progress:
                # 使用tqdm显示进度条
                with tqdm(total=len(tasks), desc="计算光学参数") as pbar:
                    for result in pool.imap_unordered(process_single_sample, tasks, chunksize=chunk_size):
                        results.append(result)
                        if not result['success']:
                            failed_count += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': len(results) - failed_count,
                            '失败': failed_count
                        })
            else:
                # 不使用进度条
                for result in pool.imap_unordered(process_single_sample, tasks, chunksize=chunk_size):
                    results.append(result)
                    if not result['success']:
                        failed_count += 1
        
        # 添加无效样本的结果
        for sample in invalid_samples:
            error_result = {
                'set_id': sample.get('set_id', 0),
                'rm_count': sample.get('rm_count', 0),
                'rm_values': sample.get('rm_values', []),
                'success': False,
                'error_message': sample.get('validation_error', '数据验证失败'),
                'focal_length': 0.0,
                'compression_ratio': 0.0,
                'sidelobe_ratio': 0.0,
                'Rm': 0.0,
                'Im': 0.0,
                'I0': 0.0
            }
            results.append(error_result)
        
        # 按set_id排序
        results.sort(key=lambda x: x['set_id'])
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """
        保存计算结果
        
        参数:
        results: 计算结果列表
        output_file: 输出文件路径
        """
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 将rm_values转换为字符串
        df['rm_values'] = df['rm_values'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        
        # 保存到CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 生成统计报告
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"\n计算结果统计:")
        print(f"  总样本数: {len(results)}")
        print(f"  成功计算: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
        print(f"  计算失败: {len(failed_results)} ({len(failed_results)/len(results)*100:.1f}%)")
        
        if successful_results:
            focal_lengths = [r['focal_length'] for r in successful_results]
            compression_ratios = [r['compression_ratio'] for r in successful_results]
            sidelobe_ratios = [r['sidelobe_ratio'] for r in successful_results]
            
            print(f"\n成功样本的统计信息:")
            print(f"  焦距范围: {min(focal_lengths):.2f} - {max(focal_lengths):.2f} mm")
            print(f"  压缩比范围: {min(compression_ratios):.4f} - {max(compression_ratios):.4f}")
            print(f"  旁瓣比范围: {min(sidelobe_ratios):.4f} - {max(sidelobe_ratios):.4f}")
        
        if failed_results:
            print(f"\n失败样本的错误类型:")
            error_types = {}
            for result in failed_results:
                error_msg = result['error_message']
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count} 个")


def objective_function(rm_values: np.ndarray) -> float:
    """
    物镜函数 (或成本函数).
    
    参数:
    rm_values: 一个包含20个半径值的一维numpy数组.
    
    返回:
    cost: 一个浮点数，表示当前这组rm_values的“成本”，越小越好.
    """
    # --- 1. 数据准备和约束检查 ---
    # 差分进化算法的输入是numpy数组，我们先将其转换为列表
    rm_list = sorted(list(rm_values)) # 确保半径是排序的
    
    # 创建一个模拟的 "sample" 字典，以适配您现有的代码结构
    sample = {
        'set_id': 0, # 临时ID
        'rm_count': len(rm_list),
        'rm_values': rm_list
    }
    
    # 创建一个模拟的 "config" 字典
    config = ParallelOpticalCalculator.get_default_config()
    
    # --- 2. 调用您的计算代码 ---
    # 为了简化，我们直接调用核心处理函数 process_single_sample
    # 注意：在优化循环中，并行是多余的，所以我们直接调用单个处理函数
    try:
        result = process_single_sample((sample, config))
        
        if not result['success']:
            # 如果计算失败，返回一个很大的成本值（惩罚项）
            return 1e6 
        
        fwhm_lambda = result['compression_ratio'] # FWHM / λ
        sidelobe_ratio = result['sidelobe_ratio']
        
        # --- 3. 定义成本函数 ---
        # 目标: FWHM < 0.4λ, Sidelobe Ratio < 0.2
        target_fwhm = 0.4
        target_sidelobe = 0.2
        
        # 定义权重，可以根据您的偏好调整
        w1 = 0.6  # FWHM的权重
        w2 = 0.4  # 旁瓣比的权重
        
        # 计算成本。我们只惩罚超出目标的部分
        cost1 = max(0, fwhm_lambda - target_fwhm)
        cost2 = max(0, sidelobe_ratio - target_sidelobe)
        
        # 一个更高级的成本函数，如果达到目标，则继续优化以追求极致
        if fwhm_lambda < target_fwhm and sidelobe_ratio < target_sidelobe:
            # 如果两个目标都已达到，我们的新目标是让它们尽可能小
            cost = w1 * fwhm_lambda + w2 * sidelobe_ratio
        else:
            # 如果有任何一个目标未达到，我们惩罚超出部分
            # 加上一个较大的基数以区分已达标和未达标的情况
            cost = 1.0 + w1 * max(0, fwhm_lambda - target_fwhm) + w2 * max(0, sidelobe_ratio - target_sidelobe)

        # 打印中间过程，方便观察
        print(f"尝试: FWHM={fwhm_lambda:.4f}λ, Sidelobe={sidelobe_ratio:.4f}, Cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        # 任何异常都返回一个很大的成本
        # print(f"计算中出现异常: {e}") # 调试时可以打开
        return 1e6
    
# ----------------------------------------------------------------
# 修复后的解决方案
# ----------------------------------------------------------------

def objective_function_diff(d_values: np.ndarray) -> float:
    """
    修复后的物镜函数，处理差值并确保半径在有效范围内
    """
    # --- 1. 从差值重构半径值 ---
    rm_values = np.cumsum(d_values)
    
    # --- 2. 更严格的边界检查 ---
    r_max = 25.0
    min_radius = 0.1  # 最小半径
    
    # 检查所有半径是否在有效范围内
    if rm_values[-1] > r_max or any(r < min_radius for r in rm_values):
        # 返回与超出程度相关的惩罚，而不是固定的大值
        max_exceed = max(0, rm_values[-1] - r_max)
        min_violation = max(0, min_radius - np.min(rm_values))
        penalty = 1000 * (max_exceed + min_violation)
        return 1e4 + penalty
    
    # 检查半径是否严格递增
    if any(rm_values[i] >= rm_values[i+1] for i in range(len(rm_values)-1)):
        return 1e4
    
    # --- 3. 调用光学计算 ---
    rm_list = list(rm_values)
    
    sample = {'set_id': 0, 'rm_count': len(rm_list), 'rm_values': rm_list}
    config = ParallelOpticalCalculator.get_default_config()
    
    try:
        result = process_single_sample((sample, config))
        if not result['success']:
            return 1e4  # 计算失败惩罚
            
        fwhm_lambda = result['compression_ratio']
        sidelobe_ratio = result['sidelobe_ratio']
        
        target_fwhm = 0.4
        target_sidelobe = 0.2
        w1 = 0.6
        w2 = 0.4
        
        # 改进的成本函数
        if fwhm_lambda < target_fwhm and sidelobe_ratio < target_sidelobe:
            # 两个目标都达到，追求更优
            cost = w1 * fwhm_lambda + w2 * sidelobe_ratio
        else:
            # 未达到目标，惩罚超出部分
            cost = 1.0 + w1 * max(0, fwhm_lambda - target_fwhm) + w2 * max(0, sidelobe_ratio - target_sidelobe)

        print(f"尝试: FWHM={fwhm_lambda:.4f}λ, Sidelobe={sidelobe_ratio:.4f}, Cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        return 1e4

# --- 修复边界设置 ---
def create_bounds(num_params=20, r_max=25.0):
    """
    创建更合理的边界
    """
    bounds = []
    
    # 第一个差值（即第一个半径）的范围
    bounds.append((0.1, r_max / num_params * 2))
    
    # 中间差值：确保累积和不会太快超过r_max
    for i in range(1, num_params):
        # 随着参数增加，允许的差值逐渐减小
        max_diff = (r_max * 0.8) / (num_params - i)  # 保守估计
        bounds.append((0.05, max_diff))
    
    return bounds

# --- 替代方案：直接优化半径值 ---
def objective_function_direct(rm_values: np.ndarray) -> float:
    """
    直接优化半径值的替代方案
    """
    # 检查边界
    r_max = 25.0
    if any(r > r_max or r < 0 for r in rm_values):
        # 返回与超出程度相关的惩罚
        penalty = sum(max(0, r - r_max) for r in rm_values) + sum(max(0, -r) for r in rm_values)
        return 1e4 + penalty * 100
    
    # 检查是否严格递增
    if any(rm_values[i] >= rm_values[i+1] for i in range(len(rm_values)-1)):
        return 1e4
    
    rm_list = list(rm_values)
    sample = {'set_id': 0, 'rm_count': len(rm_list), 'rm_values': rm_list}
    config = ParallelOpticalCalculator.get_default_config()
    
    try:
        result = process_single_sample((sample, config))
        if not result['success']:
            return 1e4
            
        fwhm_lambda = result['compression_ratio']
        sidelobe_ratio = result['sidelobe_ratio']
        
        target_fwhm = 0.4
        target_sidelobe = 0.2
        w1 = 0.6
        w2 = 0.4
        
        if fwhm_lambda < target_fwhm and sidelobe_ratio < target_sidelobe:
            cost = w1 * fwhm_lambda + w2 * sidelobe_ratio
        else:
            cost = 1.0 + w1 * max(0, fwhm_lambda - target_fwhm) + w2 * max(0, sidelobe_ratio - target_sidelobe)

        print(f"尝试: FWHM={fwhm_lambda:.4f}λ, Sidelobe={sidelobe_ratio:.4f}, Cost={cost:.4f}")
        
        return cost
        
    except Exception as e:
        return 1e4

# --- 运行修复后的优化 ---
if __name__ == "__main__":
    mp.freeze_support()
    
    print("开始使用修复后的差分进化算法进行优化...")
    
    # 方法1：使用差值方法（修复后）
    print("方法1：使用差值参数化...")
    bounds_diff = create_bounds(20, 25.0)
    
    # 先进行小规模测试
    result = differential_evolution(
        func=objective_function_diff, 
        bounds=bounds_diff, 
        strategy='best1bin', 
        maxiter=500,          # 先用较少的迭代测试
        popsize=20,          # 较小的种群
        tol=0.01, 
        mutation=(0.5, 1), 
        recombination=0.7,
        disp=True,
        workers=-1,
        updating='deferred'
    )
    
    if result.fun < 1e4:  # 如果找到了有效解
        print("\n方法1优化成功!")
        optimal_d_values = result.x
        optimal_rm_values = np.cumsum(optimal_d_values)
        print("最优环带半径:", optimal_rm_values)
    else:
        print("\n方法1未找到有效解，尝试方法2...")
        
        # 方法2：直接优化半径
        print("方法2：直接优化半径值...")
        bounds_direct = [(0.1, 25.0)] * 20
        
        result = differential_evolution(
            func=objective_function_direct, 
            bounds=bounds_direct, 
            strategy='best1bin', 
            maxiter=50,
            popsize=10,
            tol=0.01, 
            mutation=(0.5, 1), 
            recombination=0.7,
            disp=True,
            workers=-1,
            updating='deferred'
        )
        
        if result.fun < 1e4:
            print("\n方法2优化成功!")
            optimal_rm_values = result.x
            print("最优环带半径:", optimal_rm_values)
        else:
            print("\n两种方法都未找到有效解，需要调整参数或检查计算逻辑")
    
    # 最终验证
    if result.fun < 1e4:
        print("\n对最优解进行最终验证...")
        final_sample = {'set_id': 'optimal', 'rm_count': 20, 'rm_values': list(optimal_rm_values)}
        final_config = ParallelOpticalCalculator.get_default_config()
        
        final_metrics = process_single_sample((final_sample, final_config))
        
        if final_metrics['success']:
            print("-------------------------------------------")
            print("最终性能指标:")
            print(f"  焦距: {final_metrics['focal_length']:.4f} mm")
            print(f"  半高全宽: {final_metrics['Rm']:.4f} mm")
            print(f"  压缩比: {final_metrics['compression_ratio']:.4f}")
            print(f"  旁瓣比: {final_metrics['sidelobe_ratio']:.4f}")
            print("-------------------------------------------")
            
            if final_metrics['compression_ratio'] < 0.4 and final_metrics['sidelobe_ratio'] < 0.2:
                print("🎉 成功找到了超越目标的设计！")
            else:
                print("提示：接近目标，可以增加迭代次数进一步优化。")
        else:
            print("最终验证失败:", final_metrics.get('error_message', '未知错误'))