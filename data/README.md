# data

完整数据集不随仓库提交。可以使用下面的命令生成小样本数据：

```powershell
python scripts/generate_rm_dataset.py --num-samples 100 --output-dir data/rm_dataset
```

如需生成 v2 数据集：

```powershell
python scripts/generate_rm_dataset_v2.py --num-samples 100 --output-dir data/rm_dataset_v2
```
