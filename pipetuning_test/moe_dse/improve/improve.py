import pandas as pd
import numpy as np
import os
import re
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

# 存储详细结果
detail_results = []

# 存储每种硬件的汇总结果
hardware_summary = {}

# 处理每个硬件文件
for hardware, filename in hardware_files.items():
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    if not os.path.exists(filepath):
        print(f"警告: 文件 {filepath} 不存在，跳过")
        continue
    
    print(f"正在处理 {hardware}...")
    
    # 读取CSV文件
    df = pd.read_csv(filepath)
    
    # 按输入参数分组
    grouped = df.groupby(param_cols)
    
    # 存储当前硬件的所有提升数据（用于计算平均值）
    median_improvements = []
    mean_improvements = []
    
    # 处理每个参数组合
    for params, group in grouped:
        weight_type, M, E, topk, H, N = params
        
        # 获取该参数组合下的所有avg_duration
        durations = group['avg_duration'].values
        
        # 计算最小值、中位数、平均值
        min_duration = np.min(durations)
        median_duration = np.median(durations)
        mean_duration = np.mean(durations)
        
        # 计算提升百分比
        # 提升 = (原值 - 最优值) / 原值 * 100%
        # 或 = (1 - 最优值/原值) * 100%
        if median_duration > 0:
            median_improvement = (1 - min_duration / median_duration) * 100
        else:
            median_improvement = 0.0
        
        if mean_duration > 0:
            mean_improvement = (1 - min_duration / mean_duration) * 100
        else:
            mean_improvement = 0.0
        
        # 记录详细结果
        detail_results.append({
            '硬件': hardware,
            'weight_type': weight_type,
            'M': M,
            'E': E,
            'topk': topk,
            'H': H,
            'N': N,
            '相比中位数提升(%)': round(median_improvement, 4),
            '相比平均数提升(%)': round(mean_improvement, 4)
        })
        
        # 添加到汇总列表
        median_improvements.append(median_improvement)
        mean_improvements.append(mean_improvement)
    
    # 计算当前硬件的中位提升/平均提升
    avg_median_improvement = np.median(median_improvements) if median_improvements else 0.0
    avg_mean_improvement = np.mean(mean_improvements) if mean_improvements else 0.0
    
    hardware_summary[hardware] = {
        '相比中位数提升(%)': round(avg_median_improvement, 4),
        '相比平均数提升(%)': round(avg_mean_improvement, 4)
    }
    
    print(f"  {hardware} 处理完成，共有 {len(median_improvements)} 个参数组合")

# 转换为DataFrame并保存详细结果
detail_df = pd.DataFrame(detail_results)
detail_output_path = os.path.join(os.path.dirname(__file__), 'improvement_detail.csv')
detail_df.to_csv(detail_output_path, index=False, encoding='utf-8-sig')
print(f"\n详细结果已保存到: {detail_output_path}")
print(f"共 {len(detail_df)} 条记录")

# 转换为DataFrame并保存汇总结果
summary_results = []
for hardware in hardware_summary:
    summary_results.append({
        '硬件': hardware,
        '相比中位数提升(%)': hardware_summary[hardware]['相比中位数提升(%)'],
        '相比平均数提升(%)': hardware_summary[hardware]['相比平均数提升(%)']
    })

summary_df = pd.DataFrame(summary_results)
summary_output_path = os.path.join(os.path.dirname(__file__), 'improvement_sum.csv')
summary_df.to_csv(summary_output_path, index=False, encoding='utf-8-sig')
print(f"\n汇总结果已保存到: {summary_output_path}")
print(f"共 {len(summary_df)} 条记录")

# 打印汇总信息
print("\n汇总结果预览:")
print(summary_df.to_string(index=False))

# 绘制提升百分比分布图
print("\n正在绘制提升百分比分布图...")

# 创建两个图表：一个显示相比中位数提升的分布，一个显示相比平均数提升的分布
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
fig1.suptitle('Improvement Distribution vs Median', fontsize=16, fontweight='bold')

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Improvement Distribution vs Mean', fontsize=16, fontweight='bold')

# 展平axes数组以便迭代
axes1 = axes1.flatten()
axes2 = axes2.flatten()

# 定义百分比区间（bin）
bin_width = 2  # 每个区间宽度为2%
max_percent = 100
bins = np.arange(0, max_percent + bin_width, bin_width)

# 为每个硬件绘制分布图
for idx, hardware in enumerate(sorted(hardware_files.keys())):
    # 获取当前硬件的详细数据
    hardware_data = detail_df[detail_df['硬件'] == hardware]
    
    # 相比中位数提升的分布
    median_improvements = hardware_data['相比中位数提升(%)'].values
    
    axes1[idx].hist(median_improvements, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    axes1[idx].set_title(f'{hardware}', fontsize=12, fontweight='bold')
    axes1[idx].set_xlabel('Improvement Percentage Range (%)', fontsize=10)
    axes1[idx].set_ylabel('Number of Configurations', fontsize=10)
    axes1[idx].grid(True, alpha=0.3, axis='y')
    axes1[idx].set_xlim(left=0)
    
    # 添加统计信息文本
    median_mean = np.mean(median_improvements)
    median_median = np.median(median_improvements)
    axes1[idx].text(0.05, 0.95, f'Mean: {median_mean:.2f}%\nMedian: {median_median:.2f}%', 
                    transform=axes1[idx].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 相比平均数提升的分布
    mean_improvements = hardware_data['相比平均数提升(%)'].values
    
    axes2[idx].hist(mean_improvements, bins=bins, edgecolor='black', alpha=0.7, color='lightcoral')
    axes2[idx].set_title(f'{hardware}', fontsize=12, fontweight='bold')
    axes2[idx].set_xlabel('Improvement Percentage Range (%)', fontsize=10)
    axes2[idx].set_ylabel('Number of Configurations', fontsize=10)
    axes2[idx].grid(True, alpha=0.3, axis='y')
    axes2[idx].set_xlim(left=0)
    
    # 添加统计信息文本
    mean_mean = np.mean(mean_improvements)
    mean_median = np.median(mean_improvements)
    axes2[idx].text(0.05, 0.95, f'Mean: {mean_mean:.2f}%\nMedian: {mean_median:.2f}%', 
                    transform=axes2[idx].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# 保存图表
plot_output_path1 = os.path.join(os.path.dirname(__file__), 'improvement_distribution_median.png')
plot_output_path2 = os.path.join(os.path.dirname(__file__), 'improvement_distribution_mean.png')

fig1.savefig(plot_output_path1, dpi=300, bbox_inches='tight')
fig2.savefig(plot_output_path2, dpi=300, bbox_inches='tight')

print(f"Improvement distribution vs median saved to: {plot_output_path1}")
print(f"Improvement distribution vs mean saved to: {plot_output_path2}")

plt.close('all')

