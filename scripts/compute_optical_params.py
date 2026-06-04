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
warnings.filterwarnings('ignore')

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
                         z_range: Tuple[float, float] = (10, 100),
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

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算光学参数')
    parser.add_argument('--input', '-i', default='data/rm_dataset/train_dataset.csv', 
                       help='输入数据集文件路径')
    parser.add_argument('--output', '-o', default='outputs/optical_params_results.csv', 
                       help='输出结果文件路径')
    parser.add_argument('--processes', '-p', type=int, default=None,
                       help='并行进程数 (默认: CPU核心数-1)')
    parser.add_argument('--no-progress', action='store_true',
                       help='不显示进度条')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print("开始光学参数计算...")
    start_time = time.time()
    
    try:
        # 读取输入数据
        print(f"读取输入文件: {args.input}")
        df = pd.read_csv(args.input)
        dataset = df.to_dict('records')
        
        # 配置并行计算器
        config = ParallelOpticalCalculator.get_default_config()
        if args.processes:
            config['parallel']['num_processes'] = args.processes
        
        calculator = ParallelOpticalCalculator(config)
        
        # 计算光学参数
        results = calculator.compute_optical_params(
            dataset, 
            show_progress=not args.no_progress
        )
        
        # 保存结果
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        calculator.save_results(results, args.output)
        
        end_time = time.time()
        print(f"\n计算完成! 总耗时: {end_time - start_time:.2f} 秒")
        print(f"结果保存到: {args.output}")
        
    except Exception as e:
        print(f"计算过程中发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # 确保在Windows下正确使用multiprocessing
    mp.freeze_support()
    main()
