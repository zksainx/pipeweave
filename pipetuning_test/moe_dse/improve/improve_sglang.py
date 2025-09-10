import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib使用默认字体（避免中文字体问题）
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 定义硬件文件和对应的硬件名称
hardware_files = {
    'A40': 'moe_dse_A40.csv',
    'A100': 'moe_dse_A100.csv',
    'H20': 'moe_dse_H20.csv',
    'H800': 'moe_dse_H800.csv',
    'L40': 'moe_dse_L40.csv',
    'PRO6000S': 'moe_dse_PRO6000S.csv'
}

# 输入参数列
param_cols = ['weight_type', 'M', 'E', 'topk', 'H', 'N']

# SGLANG 启发式参数选择策略
def get_sglang_config(M, E):
    """
    根据 M 和 E 的值返回 sglang 启发式参数配置
    """
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 16,
            "num_warps": 4,
            "num_stages": 3,
        }

# 存储详细结果
detail_results = []

# 存储每种硬件的汇总结果（用于后续分析）
hardware_improvements = {}

# 处理每个硬件文件
for hardware, filename in hardware_files.items():
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist, skipping...")
        continue
    
    print(f"Processing {hardware}...")
    
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 按输入参数分组
    grouped = df.groupby(param_cols)
    
    # 存储当前硬件的所有提升数据
    improvements = []
    
    # 处理每个参数组合
    for params, group in grouped:
        weight_type, M, E, topk, H, N = params
        
        # 获取该参数组合下的最小 avg_duration
        min_duration = group['avg_duration'].min()
        
        # 获取 sglang 启发式配置
        sglang_config = get_sglang_config(M, E)
        
        # 在数据中查找匹配 sglang 配置的行
        sglang_mask = (
            (group['BLOCK_SIZE_M'] == sglang_config['BLOCK_SIZE_M']) &
            (group['BLOCK_SIZE_N'] == sglang_config['BLOCK_SIZE_N']) &
            (group['BLOCK_SIZE_K'] == sglang_config['BLOCK_SIZE_K']) &
            (group['GROUP_SIZE_M'] == sglang_config['GROUP_SIZE_M']) &
            (group['num_warps'] == sglang_config['num_warps']) &
            (group['num_stages'] == sglang_config['num_stages'])
        )
        
        sglang_rows = group[sglang_mask]
        
        if len(sglang_rows) == 0:
            # 如果找不到完全匹配的配置，跳过这个参数组合
            print(f"  Warning: No matching sglang config for {params} in {hardware}, skipping...")
            continue
        
        # 获取 sglang 配置的 avg_duration（如果有多个，取平均值）
        sglang_duration = sglang_rows['avg_duration'].mean()
        
        # 计算提升百分比
        # 提升 = (sglang_duration - min_duration) / sglang_duration * 100%
        if sglang_duration > 0:
            improvement = (1 - min_duration / sglang_duration) * 100
        else:
            improvement = 0.0
        
        # 记录详细结果
        detail_results.append({
            '硬件': hardware,
            'weight_type': weight_type,
            'M': M,
            'E': E,
            'topk': topk,
            'H': H,
            'N': N,
            '相比sglang提升(%)': round(improvement, 4)
        })
        
        # 添加到汇总列表
        improvements.append(improvement)
    
    # 存储当前硬件的提升数据
    hardware_improvements[hardware] = improvements
    
    print(f"  {hardware} processing completed, {len(improvements)} parameter combinations")

# 转换为DataFrame并保存详细结果
detail_df = pd.DataFrame(detail_results)
detail_output_path = os.path.join(os.path.dirname(__file__), 'improvement_sglang_detail.csv')
detail_df.to_csv(detail_output_path, index=False, encoding='utf-8-sig')
print(f"\nDetailed results saved to: {detail_output_path}")
print(f"Total {len(detail_df)} records")

# 绘制提升百分比分布图
print("\nDrawing improvement percentage distribution chart...")

# 创建图表：显示相比sglang提升的分布
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Improvement Distribution vs SGLANG Heuristic', fontsize=16, fontweight='bold')

# 展平axes数组以便迭代
axes = axes.flatten()

# 定义百分比区间（bin）
bin_width = 2  # 每个区间宽度为2%
max_percent = 100
bins = np.arange(0, max_percent + bin_width, bin_width)

# 为每个硬件绘制分布图
for idx, hardware in enumerate(sorted(hardware_files.keys())):
    if hardware not in hardware_improvements:
        continue
    
    # 获取当前硬件的详细数据
    hardware_data = detail_df[detail_df['硬件'] == hardware]
    
    if len(hardware_data) == 0:
        continue
    
    improvements = hardware_data['相比sglang提升(%)'].values
    
    axes[idx].hist(improvements, bins=bins, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[idx].set_title(f'{hardware}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Improvement Percentage Range (%)', fontsize=10)
    axes[idx].set_ylabel('Number of Configurations', fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_xlim(left=0)
    
    # 添加统计信息文本
    improvement_mean = np.mean(improvements)
    improvement_median = np.median(improvements)
    axes[idx].text(0.05, 0.95, f'Mean: {improvement_mean:.2f}%\nMedian: {improvement_median:.2f}%', 
                    transform=axes[idx].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图表
plot_output_path = os.path.join(os.path.dirname(__file__), 'improvement_sglang.png')
fig.savefig(plot_output_path, dpi=300, bbox_inches='tight')

print(f"Improvement distribution vs SGLANG saved to: {plot_output_path}")

plt.close('all')

print("\nProcessing completed!")

