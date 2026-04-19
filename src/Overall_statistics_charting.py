# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from tqdm import tqdm
import matplotlib.font_manager as fm
import platform
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 参数配置区
# ==============================================================================
# 输入和输出的根目录保持一致
INPUT_FOLDER = r'data\Overall_data_statistics\Roman_data'
OUTPUT_ROOT_FOLDER = r'data\Overall_data_statistics\Roman_data'

# 图表参数
FIG_SIZE_DIST = (20, 15)  # 基础分布图尺寸
FIG_SIZE_MSD = (16, 6)    # 扩散分析图尺寸
DPI = 300
REMOVE_OUTLIERS_QUANTILE = 0.995 # 基础统计图过滤极值
REBOUND_THRESHOLD = 0.2     # 截断阈值

# MSD计算参数
TIME_INTERVAL_STEP = 60
MAX_TIME_INTERVAL = 7200
MIN_TRAJ_POINTS = 10
MIN_SAMPLES = 30

# ==============================================================================
# 🛠️ 字体与样式
# ==============================================================================
def fix_chinese_font():
    system_name = platform.system()
    font_path = None
    if system_name == "Windows":
        candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
        for path in candidates:
            if os.path.exists(path): font_path = path; break
    elif system_name == "Darwin":
        candidates = ["/System/Library/Fonts/PingFang.ttc", "/Library/Fonts/Arial Unicode.ttf"]
        for path in candidates:
            if os.path.exists(path): font_path = path; break
    
    if font_path:
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = my_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        return my_font.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        return 'sans-serif'

FONT_NAME = fix_chinese_font()
sns.set_style("whitegrid", {"grid.linestyle": "--"})

# ==============================================================================
# 🧮 通用工具
# ==============================================================================
def haversine_np(lon1, lat1, lon2, lat2):
    """向量化计算距离"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * 1000 * c

def ensure_datetime(df, col='time'):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# ==============================================================================
# 📊 图表1: 基础统计分布概览 (四合一)
# ==============================================================================
def get_basic_stats(df):
    # 兼容性检查：确保列名存在且未反转
    if 'lat' in df.columns and df['lat'].abs().max() > 90:
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})
    
    df = ensure_datetime(df)
    
    displacements, durations, flights, speeds = [], [], [], []
    for _, group in df.groupby('id'):
        if len(group) < 2: continue
        lats, lons = group['lat'].values, group['lon'].values
        times = group['time'].values.astype(np.int64) // 10**9
        
        # 轨迹级指标
        d = haversine_np(lons[0], lats[0], lons[-1], lats[-1])
        t = (times[-1] - times[0]) / 60.0 # minutes
        if t > 0: 
            displacements.append(d)
            durations.append(t)
            
        # 步级指标 (Step-wise)
        d_steps = haversine_np(lons[:-1], lats[:-1], lons[1:], lats[1:])
        t_steps = times[1:] - times[:-1]
        valid = t_steps > 0
        v_dist = d_steps[valid]
        v_time = t_steps[valid]
        
        flights.extend(v_dist)
        speeds.extend((v_dist / v_time) * 3.6) # km/h

    return displacements, durations, flights, speeds

def draw_dist_subplot(ax, data, title, xlabel, unit, color):
    if not data: return
    s = pd.Series(data).dropna()
    s = s[~s.isin([np.inf, -np.inf])]
    if len(s) == 0: return
    
    # 过滤极值
    limit = s.quantile(REMOVE_OUTLIERS_QUANTILE)
    plot_data = s[s <= limit]
    
    # 绘图
    sns.histplot(plot_data, bins=50, kde=True, ax=ax, color=color, alpha=0.7, edgecolor='white')
    
    # 统计框
    stats_text = (f'样本数: {len(s):,}\n'
                  f'均值: {s.mean():.2f} {unit}\n'
                  f'中位数: {s.median():.2f} {unit}\n'
                  f'最大值: {s.max():.2f} {unit}')
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontname=FONT_NAME)
    
    ax.set_title(title, fontsize=16, fontweight='bold', fontname=FONT_NAME)
    ax.set_xlabel(f"{xlabel} ({unit})", fontsize=12, fontname=FONT_NAME)
    ax.set_ylabel("频数", fontsize=12, fontname=FONT_NAME)

def plot_basic_distribution(df, file_basename, output_dir):
    """生成图1: 基础统计分布"""
    disp, dur, fl, spd = get_basic_stats(df)
    if not disp: return

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_DIST)
    
    draw_dist_subplot(axes[0, 0], disp, "轨迹总位移分布", "直线距离", "m", "#3498db")
    draw_dist_subplot(axes[0, 1], dur, "轨迹总时长分布", "持续时间", "min", "#2ecc71")
    draw_dist_subplot(axes[1, 0], fl, "单次飞行长度分布", "步长", "m", "#9b59b6")
    draw_dist_subplot(axes[1, 1], spd, "瞬时速度分布", "速度", "km/h", "#e74c3c")
    
    plt.suptitle(f"基础统计分布概览: {file_basename}", fontsize=22, y=1.02, fontname=FONT_NAME)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{file_basename}_Basic_distribution_statistics.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✅ [图1] 基础分布图已保存")

# ==============================================================================
# 📊 图表2: 扩散行为相变分析 (双拼 + 截断)
# ==============================================================================
def calculate_raw_msd_alpha(df):
    """计算原始 MSD 和 Alpha"""
    df = ensure_datetime(df)
    time_intervals = np.arange(TIME_INTERVAL_STEP, MAX_TIME_INTERVAL + 1, TIME_INTERVAL_STEP)
    msd_sum = np.zeros(len(time_intervals))
    msd_count = np.zeros(len(time_intervals))
    
    for _, group in df.groupby('id'):
        if len(group) < MIN_TRAJ_POINTS: continue
        times = group['time'].values.astype(np.int64) // 10**9
        lats, lons = group['lat'].values, group['lon'].values
        
        for i, dt in enumerate(time_intervals):
            for j in range(len(times) - 1):
                target = times[j] + dt
                # 二分查找
                idx = np.searchsorted(times, target)
                best_idx, min_diff = -1, 61
                for k in [idx-1, idx]:
                    if 0 <= k < len(times) and abs(times[k] - target) < min_diff:
                        min_diff = abs(times[k] - target)
                        best_idx = k
                if best_idx != -1:
                    d = haversine_np(lons[j], lats[j], lons[best_idx], lats[best_idx])
                    msd_sum[i] += d**2
                    msd_count[i] += 1
    
    t_list, msd_list = [], []
    for k in range(len(time_intervals)):
        if msd_count[k] >= MIN_SAMPLES:
            t_list.append(time_intervals[k])
            msd_list.append(msd_sum[k] / msd_count[k])
            
    t_arr, msd_arr = np.array(t_list), np.array(msd_list)
    if len(t_arr) < 5: return None, None, None
    
    # Alpha
    log_t, log_msd = np.log(t_arr), np.log(msd_arr)
    win = min(7, len(log_t))
    if win % 2 == 0: win -= 1
    if win < 3: win = 3
    try:
        smooth_log = savgol_filter(log_msd, win, 2)
        alpha_arr = np.gradient(smooth_log, log_t)
    except:
        alpha_arr = np.full_like(t_arr, np.nan)
        
    return t_arr, msd_arr, alpha_arr

def strict_rebound_cutoff(t, msd, alpha):
    """触底反弹及到达0严格截断"""
    if len(alpha) < 5: return t, msd, alpha, "TooShort", len(alpha)
    
    running_min = alpha[0]
    cutoff_idx = len(alpha)
    reason = "Complete"
    
    for i in range(len(alpha)):
        if alpha[i] < running_min: running_min = alpha[i]
        
        # 截断条件1: 曾进入亚扩散区且反弹超过阈值
        if running_min < 1.0 and (alpha[i] > running_min + REBOUND_THRESHOLD):
            cutoff_idx = i
            reason = f"Rebound > {REBOUND_THRESHOLD} (Min={running_min:.2f})"
            break
            
        # 截断条件2: Alpha 到达或低于 0
        if alpha[i] <= 0:
            cutoff_idx = i
            reason = "Alpha <= 0"
            break
            
    return t[:cutoff_idx], msd[:cutoff_idx], alpha[:cutoff_idx], reason, cutoff_idx

def plot_vis_result(t, msd, alpha, file_basename, output_dir):
    """生成图2: Analysis_of_diffusion_phase_transformation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_MSD)

    # 左图 MSD
    ax1.plot(t, msd, 'o-', color='tab:blue', linewidth=1.5, markersize=4, alpha=0.8, label='MSD')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('时间间隔 $\Delta t$ (s)', fontname=FONT_NAME, fontsize=12)
    ax1.set_ylabel('均方位移 MSD ($m^2$)', fontname=FONT_NAME, fontsize=12)
    ax1.set_title('均方位移演化', fontname=FONT_NAME, fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.3)

    # 右图 Alpha
    ax2.plot(t, alpha, 'o-', color='tab:blue', linewidth=1.5, markersize=4, alpha=0.8, label='$\\alpha(t)$')
    
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='正常扩散')
    ax2.fill_between(t, 0, 0.5, color='red', alpha=0.1, label='亚扩散')
    ax2.fill_between(t, 0.5, 1.0, color='yellow', alpha=0.1)
    ax2.fill_between(t, 1.0, 2.0, color='lightgreen', alpha=0.1, label='超扩散')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('时间间隔 $\Delta t$ (s)', fontname=FONT_NAME, fontsize=12)
    ax2.set_ylabel('扩散指数 $\\alpha(t)$', fontname=FONT_NAME, fontsize=12)
    ax2.set_title('扩散行为相变分析', fontname=FONT_NAME, fontsize=14, fontweight='bold')
    
    if len(alpha) > 0:
        y_min, y_max = np.min(alpha), np.max(alpha)
        ax2.set_ylim(max(0, y_min-0.2), min(3.0, y_max+0.2))

    ax2.legend(prop={'family': FONT_NAME})
    ax2.grid(True, which="both", linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_basename}_Analysis_of_diffusion_phase_transformation.png"), dpi=DPI, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 🚀 主程序
# ==============================================================================
def process_single_file(file_path):
    # 获取无后缀的文件名
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📄 处理文件: {file_basename}")
    
    # 【核心修改】：创建同名子文件夹
    current_output_dir = os.path.join(OUTPUT_ROOT_FOLDER, file_basename)
    if not os.path.exists(current_output_dir):
        os.makedirs(current_output_dir)
        print(f"  📁 创建专属文件夹: {current_output_dir}")
    else:
        print(f"  📁 使用现有文件夹: {current_output_dir}")

    try:
        try: df = pd.read_csv(file_path, encoding='utf-8')
        except: df = pd.read_csv(file_path, encoding='gbk')
        
        # 1. 绘制第一张图: 基础分布 (传入子文件夹路径)
        plot_basic_distribution(df, file_basename, current_output_dir)
        
        # 2. 计算 MSD 全量
        t_raw, msd_raw, alpha_raw = calculate_raw_msd_alpha(df)
        if t_raw is None: 
            print("  ⚠️ 数据点不足，跳过 MSD 分析")
            return

        # 💾 保存全量数据 (保存到子文件夹)
        pd.DataFrame({
            'TimeInterval_s': t_raw, 'MSD_mean': msd_raw, 'Alpha_t': alpha_raw
        }).to_csv(os.path.join(current_output_dir, f"{file_basename}_msd_result_full.csv"), index=False, encoding='utf-8-sig')

        # 3. 截断
        t_vis, msd_vis, alpha_vis, cut_reason, cut_idx = strict_rebound_cutoff(t_raw, msd_raw, alpha_raw)
        print(f"  ✂️ 截断详情: {len(t_raw)} -> {len(t_vis)} 点 | {cut_reason}")

        # 💾 保存可视化数据 & 日志 (保存到子文件夹)
        pd.DataFrame({
            'TimeInterval_s': t_vis, 'MSD_mean': msd_vis, 'Alpha_t': alpha_vis
        }).to_csv(os.path.join(current_output_dir, f"{file_basename}_msd_result_vis.csv"), index=False, encoding='utf-8-sig')
        
        pd.DataFrame([{
            'File': file_basename, 'Original': len(t_raw), 'Vis': len(t_vis), 'Reason': cut_reason
        }]).to_csv(os.path.join(current_output_dir, f"{file_basename}_cutoff_log.csv"), index=False, encoding='utf-8-sig')

        # 4. 绘制第二张图: 扩散相变 (传入子文件夹路径)
        plot_vis_result(t_vis, msd_vis, alpha_vis, file_basename, current_output_dir)
        print(f"  ✅ [图2] 扩散分析图已保存至子文件夹")
        
    except Exception as e:
        print(f"  ❌ 出错: {e}")

if __name__ == "__main__":
    # 确保根输出目录存在
    os.makedirs(OUTPUT_ROOT_FOLDER, exist_ok=True)
    
    # 查找所有CSV文件
    # 注意：glob(..., recursive=False) 默认不递归，避免读到子文件夹里生成的结果
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    
    # 过滤掉可能存在的之前生成的结果文件（虽然现在放子文件夹了，但这层保护依然有效）
    target_files = [f for f in all_files 
                    if "msd_result" not in f and "cutoff_log" not in f and "calc_result" not in f]
    
    if not target_files:
        print(f"❌ 在 '{INPUT_FOLDER}' 未找到original_dataCSV文件")
    else:
        print(f"🚀 开始处理 {len(target_files)} 个文件...")
        for f in tqdm(target_files):
            process_single_file(f)
    print("\n🎉 全部处理完成！")