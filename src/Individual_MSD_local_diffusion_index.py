# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 2026
Combined Individual MSD Analysis (Summary Stats + Detailed Curves)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.stats import linregress
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===================================================================================
# 🔧 参数配置区
# ===================================================================================
# 输入文件夹：上一步分类后的数据
INPUT_FOLDER = r'Cleaning_data_after_segmentation\SZ其他出行数据\guang'

# 输出文件夹：你要的目标路径
OUTPUT_FOLDER = r'个体MSD分析结果\SZ其他出行数据'

# 时间间隔设置 (秒): 从60s到3600s(1小时)，步长60s
# 如果你的轨迹很长，可以适当把 3601 改大
TIME_LAGS = np.arange(60, 3601, 60) 

# 平滑参数 (用于计算局部alpha)
SMOOTH_WINDOW = 5  # 必须是奇数
POLY_ORDER = 2

# 可视化设置
PLOT_SAMPLE_NUM = 50  # 每张图随机画多少条轨迹（画太多会乱）
# ===================================================================================

def haversine_np(lon1, lat1, lon2, lat2):
    """向量化计算地球表面距离 (米)"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * 1000 * c

def analyze_trajectory(traj_df, time_lags):
    """
    核心函数：同时计算单条轨迹的统计指标和曲线数据
    """
    # 0. 基础数据准备
    t_base = traj_df['time'].astype(np.int64) // 10**9
    t_values = t_base.values
    lats = traj_df['lat'].values
    lons = traj_df['lon'].values
    
    # 基础统计
    duration = (traj_df['time'].max() - traj_df['time'].min()).total_seconds()
    if len(lats) > 1:
        dists = haversine_np(lons[:-1], lats[:-1], lons[1:], lats[1:])
        total_dist = np.sum(dists)
    else:
        total_dist = 0
        
    msd_list = []
    valid_lags = []
    
    # 1. 计算 MSD 序列
    for delta_t in time_lags:
        # 如果时间间隔比轨迹还长，就没法算了
        if delta_t >= duration: continue
        
        # 寻找对应时间点的索引
        target_times = t_values + delta_t
        idx = np.searchsorted(t_values, target_times)
        
        # 过滤无效索引
        valid_mask = idx < len(t_values)
        if not np.any(valid_mask): continue
        
        matched_idx = idx[valid_mask]
        original_idx = np.arange(len(t_values))[valid_mask]
        
        # 严格检查时间差 (容忍度 10% 或 30s)
        real_diff = np.abs(t_values[matched_idx] - t_values[original_idx] - delta_t)
        strict_mask = real_diff <= max(30, delta_t * 0.1)
        
        if not np.any(strict_mask): continue
        
        final_j = matched_idx[strict_mask]
        final_i = original_idx[strict_mask]
        
        # 计算距离平方并取平均
        d = haversine_np(lons[final_i], lats[final_i], lons[final_j], lats[final_j])
        msd_val = np.mean(d ** 2)
        
        if msd_val > 0:
            msd_list.append(msd_val)
            valid_lags.append(delta_t)
            
    # 如果有效点太少，返回空
    if len(valid_lags) < 5:
        return None, None

    # 2. 计算局部 Alpha (曲线)
    log_t = np.log(valid_lags)
    log_msd = np.log(msd_list)
    
    # 平滑处理
    curr_window = min(SMOOTH_WINDOW, len(log_msd))
    if curr_window % 2 == 0: curr_window -= 1
    if curr_window < 3: curr_window = 3
    
    try:
        log_msd_smooth = savgol_filter(log_msd, window_length=curr_window, polyorder=POLY_ORDER)
        alpha_local = np.gradient(log_msd_smooth, log_t)
    except:
        alpha_local = np.full_like(log_msd, np.nan)
        
    # 3. 计算全局 Alpha (统计值 - 线性回归)
    try:
        slope, intercept, r_value, _, _ = linregress(log_t, log_msd)
        global_alpha = slope
        diffusion_coeff = np.exp(intercept)
        r2 = r_value ** 2
    except:
        global_alpha, diffusion_coeff, r2 = np.nan, np.nan, np.nan

    # 4. 打包返回结果
    
    # A. 汇总数据 (一行)
    summary_dict = {
        'duration_sec': duration,
        'total_distance_m': total_dist,
        'avg_speed_kmh': (total_dist/duration*3.6) if duration>0 else 0,
        'global_alpha': global_alpha,       # 全局扩散指数 b
        'diffusion_coeff': diffusion_coeff, # 扩散系数 a
        'r_squared': r2,                    # 拟合优度
        'points_count': len(traj_df)
    }
    
    # B. 曲线数据 (多行 DataFrame)
    curves_df = pd.DataFrame({
        'tau': valid_lags,
        'msd': msd_list,
        'alpha_local': alpha_local
    })
    
    return summary_dict, curves_df

def plot_preview(curves_df_all, title_name, save_path):
    """绘制抽样预览图"""
    unique_ids = curves_df_all['id'].unique()
    
    # 随机抽样
    if len(unique_ids) > PLOT_SAMPLE_NUM:
        plot_ids = np.random.choice(unique_ids, PLOT_SAMPLE_NUM, replace=False)
        sub_title = f"(随机抽样 {PLOT_SAMPLE_NUM}/{len(unique_ids)})"
    else:
        plot_ids = unique_ids
        sub_title = f"(全部 {len(unique_ids)})"
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘图
    colors = plt.cm.jet(np.linspace(0, 1, len(plot_ids)))
    for uid, c in zip(plot_ids, colors):
        sub = curves_df_all[curves_df_all['id'] == uid]
        ax1.plot(sub['tau'], sub['msd'], color=c, alpha=0.3, lw=1)
        ax2.plot(sub['tau'], sub['alpha_local'], color=c, alpha=0.3, lw=1)
        
    # 设置
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'个体 MSD 曲线 {sub_title}')
    ax1.set_xlabel('时间间隔 $\\tau$ (s)')
    ax1.set_ylabel('MSD ($m^2$)')
    ax1.grid(True, alpha=0.2)
    
    ax2.set_xscale('log')
    ax2.set_title(f'个体扩散指数 $\\alpha(t)$ 曲线 {sub_title}')
    ax2.set_xlabel('时间间隔 $\\tau$ (s)')
    ax2.set_ylabel('局部 $\\alpha$')
    ax2.axhline(1, color='k', ls='--', alpha=0.5)
    ax2.set_ylim(-0.5, 2.5)
    ax2.grid(True, alpha=0.2)
    
    plt.suptitle(f"个体轨迹动力学分析 - {title_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def process_single_file(file_path, output_dir):
    filename = os.path.basename(file_path)
    file_base = os.path.splitext(filename)[0]
    print(f"\n📄 处理文件: {filename}")
    
    # 1. 读取
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            print("  ❌ 读取失败，跳过")
            return

    # 2. 清洗坐标
    if 'lat' in df.columns and df['lat'].abs().max() > 90:
        print("  ⚠️ 修复经纬度翻转")
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})
    df['time'] = pd.to_datetime(df['time'])
    
    summary_list = []
    curves_list = []
    
    # 3. 逐条计算
    # 获取Cluster ID（如果存在）
    cluster_id = None
    if 'cluster' in df.columns:
        cluster_id = df['cluster'].iloc[0] # 假设一个文件里都是同一个cluster的切片
        
    unique_ids = df['id'].unique()
    
    for uid in tqdm(unique_ids, desc="  计算个体指标", unit="traj"):
        traj = df[df['id'] == uid]
        if len(traj) < 10: continue # 忽略太短的
        
        summ, curves = analyze_trajectory(traj, TIME_LAGS)
        
        if summ is not None:
            # 添加ID信息
            summ['id'] = uid
            if cluster_id is not None: summ['cluster'] = cluster_id
            summary_list.append(summ)
            
            # 添加ID信息到曲线表
            curves['id'] = uid
            curves_list.append(curves)
            
    # 4. 保存结果
    if summary_list:
        # A. 保存汇总表
        df_summary = pd.DataFrame(summary_list)
        # 调整列顺序
        cols = ['id'] + [c for c in df_summary.columns if c != 'id']
        df_summary = df_summary[cols]
        
        path_summary = os.path.join(output_dir, f"{file_base}_summary.csv")
        df_summary.to_csv(path_summary, index=False, encoding='utf-8-sig')
        print(f"  💾 汇总表已保存: {os.path.basename(path_summary)}")
        
        # B. 保存曲线表
        df_curves = pd.concat(curves_list)
        path_curves = os.path.join(output_dir, f"{file_base}_curves.csv")
        df_curves.to_csv(path_curves, index=False, encoding='utf-8-sig')
        print(f"  💾 曲线表已保存: {os.path.basename(path_curves)}")
        
        # C. 保存预览图
        path_plot = os.path.join(output_dir, f"{file_base}_plot.png")
        plot_preview(df_curves, file_base, path_plot)
        print(f"  🖼️ 预览图已保存")
        
    else:
        print("  ⚠️ 未能生成有效结果")

# ===================================================================================
# 主流程
# ===================================================================================
if __name__ == "__main__":
    print("🚀 开始个体 MSD 全指标分析任务...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 搜索文件
    csv_files = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            # 只处理包含 'cluster' 的轨迹切片文件，排除汇总文件
            if file.endswith(".csv") and "cluster_" in file and "results" not in file:
                 csv_files.append(os.path.join(root, file))
                 
    if not csv_files:
        print("❌ 未找到待处理文件，请检查输入路径")
    else:
        print(f"📋 共发现 {len(csv_files)} 个文件")
        for csv in csv_files:
            process_single_file(csv, OUTPUT_FOLDER)
            
    print("\n🎉 全部处理完成！")