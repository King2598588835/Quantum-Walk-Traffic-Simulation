# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 2026
Per-File Trajectory Distribution Visualization (Fixed Chinese Fonts)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import warnings
import matplotlib.font_manager as fm
import platform

# 忽略警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 参数配置区
# ==============================================================================
INPUT_FOLDER = r'Cleaning_data_after_segmentation\SZ其他出行数据\guang'
OUTPUT_FOLDER = r'统计分布图表\SZ其他出行数据\分图合集'

# 过滤前 0.5% 的极值
REMOVE_OUTLIERS_QUANTILE = 0.995  
FIG_SIZE = (20, 15)
DPI = 300
# ==============================================================================

# ------------------------------------------------------------------------------
# 🛠️ 字体强制修复模块 (暴力解决小方框问题)
# ------------------------------------------------------------------------------
def fix_chinese_font():
    """
    尝试直接加载 Windows/Mac 系统字体文件
    """
    system_name = platform.system()
    font_path = None
    
    # 1. 优先尝试的字体文件路径 (Windows)
    if system_name == "Windows":
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑 (首选，最好看)
            r"C:\Windows\Fonts\simhei.ttf", # 黑体 (备选)
            r"C:\Windows\Fonts\simsun.ttc", # 宋体
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
    
    # 2. Mac 系统路径
    elif system_name == "Darwin":
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf"
        ]
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break

    # 3. 应用字体
    if font_path:
        print(f"✅ 已锁定字体文件: {font_path}")
        # 加载字体
        my_font = fm.FontProperties(fname=font_path)
        # 将该字体设置为默认 sans-serif
        plt.rcParams['font.family'] = my_font.get_name()
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False 
        return my_font.get_name()
    else:
        # 4. 如果找不到文件，尝试使用通用名称 (Linux或特殊环境)
        print("⚠️ 未找到标准字体文件，尝试使用名称加载...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        return 'sans-serif'

# 初始化字体
font_family_name = fix_chinese_font()

# 设置 Seaborn 样式 (注意：要在设置完字体后应用，否则可能覆盖)
sns.set_style("whitegrid", {"grid.linestyle": "--"})
sns.set_context("talk")

# ------------------------------------------------------------------------------
# 2. 计算核心逻辑
# ------------------------------------------------------------------------------
def haversine_np(lon1, lat1, lon2, lat2):
    """向量化距离计算"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * 1000 * c

def get_stats_from_df(df):
    """计算统计指标"""
    if 'lat' in df.columns and df['lat'].abs().max() > 90:
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})
    df['time'] = pd.to_datetime(df['time'])
    
    displacements, durations, flights, speeds = [], [], [], []
    grouped = df.groupby('id')
    
    for _, group in grouped:
        if len(group) < 2: continue
        lats = group['lat'].values
        lons = group['lon'].values
        times = group['time'].values.astype(np.int64) // 10**9
        
        # 轨迹级
        d = haversine_np(lons[0], lats[0], lons[-1], lats[-1])
        t = (times[-1] - times[0]) / 60.0
        if t > 0:
            displacements.append(d)
            durations.append(t)
            
        # 步级
        d_steps = haversine_np(lons[:-1], lats[:-1], lons[1:], lats[1:])
        t_steps = times[1:] - times[:-1]
        valid = t_steps > 0
        v_dist = d_steps[valid]
        v_time = t_steps[valid]
        
        flights.extend(v_dist)
        speeds.extend((v_dist / v_time) * 3.6)
        
    return displacements, durations, flights, speeds

# ------------------------------------------------------------------------------
# 3. 绘图逻辑
# ------------------------------------------------------------------------------
def draw_subplot(ax, data, title, xlabel, unit, color):
    if not data:
        ax.text(0.5, 0.5, "无数据", ha='center', transform=ax.transAxes)
        return

    s = pd.Series(data).dropna()
    s = s[~s.isin([np.inf, -np.inf])]
    if len(s) == 0: return

    # 统计
    mean_v = s.mean()
    med_v = s.median()
    max_v = s.max()
    
    # 过滤显示范围
    limit = s.quantile(REMOVE_OUTLIERS_QUANTILE)
    plot_data = s[s <= limit]
    
    # 绘图
    sns.histplot(plot_data, bins=50, kde=True, ax=ax, color=color, alpha=0.7, edgecolor='white')
    
    # 统计框
    text_str = (f'样本数: {len(s):,}\n'
                f'均值: {mean_v:.2f} {unit}\n'
                f'中位数: {med_v:.2f} {unit}\n'
                f'最大值: {max_v:.2f} {unit}')
    
    # 确保文本框字体正常
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontname=font_family_name) # 显式指定字体
    
    ax.set_title(title, fontsize=16, fontweight='bold', fontname=font_family_name)
    ax.set_xlabel(f"{xlabel} ({unit})", fontsize=12, fontname=font_family_name)
    ax.set_ylabel("频数", fontsize=12, fontname=font_family_name)

# ------------------------------------------------------------------------------
# 4. 主程序
# ------------------------------------------------------------------------------
def process_single_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    file_base = os.path.splitext(filename)[0]
    print(f"📊 正在分析: {filename} ...")
    
    try:
        df = pd.read_csv(file_path)
        disp, dur, fl, spd = get_stats_from_df(df)
        
        if not disp:
            return

        fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE)
        
        draw_subplot(axes[0, 0], disp, "轨迹总位移分布", "直线距离", "m", "#3498db")
        draw_subplot(axes[0, 1], dur, "轨迹总时长分布", "持续时间", "min", "#2ecc71")
        draw_subplot(axes[1, 0], fl, "单次飞行长度分布", "步长", "m", "#9b59b6")
        draw_subplot(axes[1, 1], spd, "瞬时速度分布", "速度", "km/h", "#e74c3c")
        
        plt.suptitle(f"分类数据统计概览: {file_base}", fontsize=22, y=1.02, fontname=font_family_name)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"统计分布_{file_base}.png")
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 已保存: {save_path}")
        
    except Exception as e:
        print(f"  ❌ 出错: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    
    if csv_files:
        print(f"🚀 开始处理 {len(csv_files)} 个文件...")
        for f in tqdm(csv_files):
            process_single_file(f, OUTPUT_FOLDER)
    else:
        print("❌ 未找到文件")







