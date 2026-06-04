import numpy as np
import pandas as pd
import json
import os
import argparse
from typing import List, Dict, Any

class RMDatasetGenerator:
    """
    生成随机rm序列数据集的类
    用于神经网络训练数据准备
    """
    
    def __init__(self, output_dir: str = "dataset"):
        """
        初始化数据集生成器
        
        参数:
        output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 默认参数配置
        self.config = {
            'num_samples': 1000,        # 总样本数量
            'min_rm_count': 5,          # 最小rm数量
            'max_rm_count': 30,         # 最大rm数量
            'rm_precision': 0.01,       # rm精度 (mm)
            'r_max': 25.0,              # 最大半径 (mm)
            'train_ratio': 0.8,         # 训练集比例
            'val_ratio': 0.1,           # 验证集比例
            'test_ratio': 0.1,          # 测试集比例
            'random_seed': 42           # 随机种子
        }
    
    def generate_rm_sequence(self, rm_count: int) -> List[float]:
        """
        生成单个rm序列
        
        参数:
        rm_count: rm序列长度
        
        返回:
        rm_values: 排序后的rm序列
        """
        # 生成随机rm值，精度为0.01mm
        rm_values = np.random.uniform(0, self.config['r_max'], rm_count)
        rm_values = np.round(rm_values / self.config['rm_precision']) * self.config['rm_precision']
        rm_values = np.sort(rm_values)  # 升序排列
        
        return rm_values.tolist()
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """
        生成完整数据集
        
        返回:
        dataset: 包含所有样本的列表
        """
        np.random.seed(self.config['random_seed'])
        dataset = []
        
        print(f"开始生成 {self.config['num_samples']} 个rm序列样本...")
        
        for set_id in range(self.config['num_samples']):
            # 随机生成rm数量
            rm_count = np.random.randint(self.config['min_rm_count'], 
                                       self.config['max_rm_count'] + 1)
            
            # 生成rm序列
            rm_values = self.generate_rm_sequence(rm_count)
            
            # 创建样本数据
            sample = {
                'set_id': set_id + 1,
                'rm_count': rm_count,
                'rm_values': rm_values,
                'rm_mean': float(np.mean(rm_values)),
                'rm_std': float(np.std(rm_values)),
                'rm_min': float(np.min(rm_values)),
                'rm_max': float(np.max(rm_values))
            }
            
            dataset.append(sample)
            
            if (set_id + 1) % 100 == 0:
                print(f"已生成 {set_id + 1}/{self.config['num_samples']} 个样本")
        
        print("数据集生成完成!")
        return dataset
    
    def split_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        分割数据集为训练集、验证集和测试集
        
        参数:
        dataset: 完整数据集
        
        返回:
        splits: 分割后的数据集字典
        """
        np.random.shuffle(dataset)
        
        n_total = len(dataset)
        n_train = int(n_total * self.config['train_ratio'])
        n_val = int(n_total * self.config['val_ratio'])
        
        splits = {
            'train': dataset[:n_train],
            'val': dataset[n_train:n_train + n_val],
            'test': dataset[n_train + n_val:]
        }
        
        print(f"数据集分割完成:")
        print(f"  训练集: {len(splits['train'])} 个样本")
        print(f"  验证集: {len(splits['val'])} 个样本") 
        print(f"  测试集: {len(splits['test'])} 个样本")
        
        return splits
    
    def save_dataset(self, splits: Dict[str, List[Dict[str, Any]]], format: str = 'csv'):
        """
        保存数据集到文件
        
        参数:
        splits: 分割后的数据集
        format: 文件格式 ('csv' 或 'json')
        """
        for split_name, split_data in splits.items():
            if format == 'csv':
                self._save_as_csv(split_name, split_data)
            elif format == 'json':
                self._save_as_json(split_name, split_data)
        
        # 保存配置信息
        config_path = os.path.join(self.output_dir, 'dataset_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"数据集已保存到: {self.output_dir}")
    
    def _save_as_csv(self, split_name: str, data: List[Dict[str, Any]]):
        """以CSV格式保存数据集"""
        filename = os.path.join(self.output_dir, f'{split_name}_dataset.csv')
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 将rm_values列转换为字符串表示
        df['rm_values'] = df['rm_values'].apply(lambda x: json.dumps(x))
        
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"  {split_name}集已保存为: {filename}")
    
    def _save_as_json(self, split_name: str, data: List[Dict[str, Any]]):
        """以JSON格式保存数据集"""
        filename = os.path.join(self.output_dir, f'{split_name}_dataset.json')
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  {split_name}集已保存为: {filename}")
    
    def generate_statistics(self, dataset: List[Dict[str, Any]]):
        """
        生成数据集统计信息
        
        参数:
        dataset: 完整数据集
        """
        rm_counts = [sample['rm_count'] for sample in dataset]
        rm_means = [sample['rm_mean'] for sample in dataset]
        
        print("\n数据集统计信息:")
        print(f"  总样本数: {len(dataset)}")
        print(f"  rm数量范围: {min(rm_counts)} - {max(rm_counts)}")
        print(f"  rm数量均值: {np.mean(rm_counts):.2f} ± {np.std(rm_counts):.2f}")
        print(f"  rm值均值范围: {min(rm_means):.2f} - {max(rm_means):.2f} mm")
        print(f"  rm值精度: {self.config['rm_precision']} mm")
        
        # 统计rm数量分布
        count_dist = pd.Series(rm_counts).value_counts().sort_index()
        print(f"\n  rm数量分布:")
        for count, freq in count_dist.items():
            print(f"    {count}个rm: {freq}个样本 ({freq/len(dataset)*100:.1f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成 RM 训练/验证/测试数据集")
    parser.add_argument("--output-dir", default="data/rm_dataset", help="数据集输出目录")
    parser.add_argument("--num-samples", type=int, default=100000, help="生成样本数")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="输出格式")
    args = parser.parse_args()

    # 创建数据集生成器
    generator = RMDatasetGenerator(output_dir=args.output_dir)
    
    # 可选的参数调整
    generator.config.update({
        'num_samples': args.num_samples,
        'min_rm_count': 20,
        'max_rm_count': 20,
        'random_seed': args.seed,
    })
    
    # 生成数据集
    dataset = generator.generate_dataset()
    
    # 生成统计信息
    generator.generate_statistics(dataset)
    
    # 分割数据集
    splits = generator.split_dataset(dataset)
    
    # 保存数据集 (可选择CSV或JSON格式)
    generator.save_dataset(splits, format=args.format)
    
    print("\n数据集生成完成！文件说明:")
    print("  train_dataset.csv - 训练集")
    print("  val_dataset.csv   - 验证集") 
    print("  test_dataset.csv  - 测试集")
    print("  dataset_config.json - 数据集配置信息")

if __name__ == "__main__":
    main()
