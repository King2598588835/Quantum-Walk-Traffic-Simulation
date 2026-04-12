# -*- coding: utf-8 -*-
"""
Step 3: MSD Analysis with Statistical Truncation (模块化调用版)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import linregress
import warnings
import os
import glob
import platform
import matplotlib.font_manager as fm

# 忽略警告
warnings.filterwarnings("ignore")

# ===================================================================================
# 🔧 核心算法参数 (保持原样)
# ===================================================================================
TIME_INTERVALS_SEC = np.arange(60, 7201, 60) 
MIN_TRAJECTORY_POINTS = 10     
TIME_TOLERANCE = 60            
MIN_TOTAL_SAMPLES = 30         
SAVGOL_WINDOW = 7              
SAVGOL_ORDER = 2               
REBOUND_THRESHOLD = 0.2        

# -----------------------------------------------------------------------------------
# 基础工具函数 (逻辑完全保持不变)
# -----------------------------------------------------------------------------------
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
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def calculate_msd_individual_with_count(group, time_intervals, tolerance=60):
    group = group.sort_values('time').reset_index(drop=True)
    if len(group) < 2: return None
    times_sec = (group['time'].astype('int64') // 10**9).values
    lats, lons = group['lat'].values, group['lon'].values
    msd_values, sample_counts = [], []
    for delta_t in time_intervals:
        displacements_sq = []
        for i in range(len(group)):
            time_diffs = times_sec[i:] - times_sec[i]
            candidates_mask = np.abs(time_diffs - delta_t) <= tolerance
            if np.any(candidates_mask):
                best_rel_idx = np.argmin(np.abs(time_diffs[candidates_mask] - delta_t))
                best_idx = i + np.where(candidates_mask)[0][best_rel_idx]
                dist = haversine_distance(lats[i], lons[i], lats[best_idx], lons[best_idx])
                displacements_sq.append(dist ** 2)
        count = len(displacements_sq)
        msd_values.append(np.mean(displacements_sq) if count > 0 else np.nan)
        sample_counts.append(count)
    return pd.DataFrame({'TimeInterval': time_intervals, 'MSD': msd_values, 'SampleCount': sample_counts})

def fit_power_law_log_log(t, msd):
    mask = (t > 0) & (msd > 0) & (~np.isnan(msd))
    if np.sum(mask) < 5: return np.nan, np.nan, np.nan
    slope, intercept, r_val, _, _ = linregress(np.log(t[mask]), np.log(msd[mask]))
    return np.exp(intercept), slope, r_val ** 2

def compute_alpha(msd, t):
    mask = (msd > 0) & (t > 0) & (~np.isnan(msd))
    if np.sum(mask) < SAVGOL_WINDOW: return np.full_like(msd, np.nan)
    log_t, log_msd = np.log(t[mask]), np.log(msd[mask])
    actual_win = min(SAVGOL_WINDOW, len(log_msd))
    if actual_win % 2 == 0: actual_win -= 1
    if actual_win < 3: actual_win = 3
    try:
        smooth = savgol_filter(log_msd, window_length=actual_win, polyorder=SAVGOL_ORDER)
        alpha_clean = np.gradient(smooth, log_t)
        alpha = np.full_like(msd, np.nan); alpha[mask] = alpha_clean
        return alpha
    except: return np.full_like(msd, np.nan)

def strict_rebound_cutoff(t, msd, alpha):
    valid = ~np.isnan(alpha) & ~np.isnan(t)
    if np.sum(valid) < 5: return t[valid], msd[valid], alpha[valid], "Short"
    t_c, m_c, a_c = t[valid], msd[valid], alpha[valid]
    cutoff = len(a_c); r_min = a_c[0]
    for i, val in enumerate(a_c):
        if val < r_min: r_min = val
        if (r_min < 1.0 and val > r_min + REBOUND_THRESHOLD) or val <= 0:
            cutoff = i; break
    return t_c[:cutoff], m_c[:cutoff], a_c[:cutoff], "Processed"

def find_transition_point(t, alpha):
    for i in range(1, len(alpha)):
        if alpha[i] <= 1.0 and alpha[i-1] > 1.0: return t[i], alpha[i]
    return np.nan, np.nan

# -----------------------------------------------------------------------------------
# 单文件分析流程 (核心命名逻辑保持不变)
# -----------------------------------------------------------------------------------
def analyze_single_file(file_path, output_dir):
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    try: df = pd.read_csv(file_path, encoding='utf-8-sig')
    except: df = pd.read_csv(file_path, encoding='gbk')

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    msd_list = []
    for _, group in df.groupby('id'):
        if len(group) < MIN_TRAJECTORY_POINTS: continue
        res = calculate_msd_individual_with_count(group, TIME_INTERVALS_SEC, TIME_TOLERANCE)
        if res is not None: msd_list.append(res)
    
    if not msd_list: return False
    msd_grouped = pd.concat(msd_list).groupby('TimeInterval').agg(MSD_mean=('MSD', 'mean'), TotalSamples=('SampleCount', 'sum')).reset_index()
    valid_msd = msd_grouped[msd_grouped['TotalSamples'] >= MIN_TOTAL_SAMPLES]
    if len(valid_msd) < 5: return False
    
    t_raw, m_raw = valid_msd['TimeInterval'].values, valid_msd['MSD_mean'].values
    a_raw = compute_alpha(m_raw, t_raw)
    t_cut, m_cut, a_cut, _ = strict_rebound_cutoff(t_raw, m_raw, a_raw)
    if len(t_cut) < 5: return False

    cluster_name = file_base.split("轨迹")[0] if "轨迹" in file_base else file_base
    res_path = os.path.join(output_dir, f"{cluster_name}轨迹的MSD变化以及扩散指数α以及量子模拟结果.csv")
    plot_path = os.path.join(output_dir, f"{cluster_name}轨迹的MSD变化以及扩散指数α以及量子模拟结果.png")

    pd.DataFrame({'TimeInterval': t_cut, 'MSD_mean': m_cut, 'Alpha_t': a_cut}).to_csv(res_path, index=False, encoding='utf-8-sig')

    # 绘图逻辑保持原样
    fix_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.scatter(t_cut, m_cut, color='#1f77b4', alpha=0.6, s=20, label='Ensemble MSD')
    a, b, r2 = fit_power_law_log_log(t_cut, m_cut)
    if not np.isnan(b): ax1.plot(t_cut, a * t_cut ** b, 'r--', label=f'Fit: $t^{{{b:.2f}}}$ ($R^2={r2:.2f}$)')
    ax1.set_yscale('log'); ax1.set_xscale('log'); ax1.set_ylabel('MSD ($m^2$)'); ax1.legend(); ax1.grid(True, which='both', ls='--', alpha=0.2)
    ax2.plot(t_cut, a_cut, 'g.-', label=r'$\alpha(t)$')
    ax2.axhline(1.0, color='gray', ls='--'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel(r'$\alpha(t)$'); ax2.set_ylim(0, 2.5); ax2.grid(True, which='both', ls='--', alpha=0.2)
    plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close()
    return True

# -----------------------------------------------------------------------------------
# 🚀 封装的调用接口
# -----------------------------------------------------------------------------------
def run_step_3_msd_analysis(city_name):
    """
    只需调用此函数并传入城市名，即可自动执行对应的 MSD 分析
    """
    INPUT_FOLDER = os.path.join('data', '清洗分割后数据', city_name)
    OUTPUT_FOLDER = os.path.join('data', '群体msd分析结果', city_name)
    
    print(f"\n[Step 3] MSD分析启动 | 城市: {city_name}")
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 找不到输入目录: {INPUT_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    count = 0
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.endswith(".csv") and "聚类" in file and "处理数据" in file:
                rel_path = os.path.relpath(root, INPUT_FOLDER)
                current_out = os.path.join(OUTPUT_FOLDER, rel_path)
                os.makedirs(current_out, exist_ok=True)
                if analyze_single_file(os.path.join(root, file), current_out):
                    count += 1
                    print(f"   ✅ 已完成: {file}")

    print(f"🎉 {city_name} MSD分析全部完成，共处理 {count} 个聚类文件。")
