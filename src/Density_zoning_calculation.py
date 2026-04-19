# -*- coding: utf-8 -*-
"""
Step 4: Grid Density Analysis & Visualization (模块化调用版)
输入：Step 2 分类后的Cluster数据
输出：网格密度分布 CSV 与 可视化图
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.font_manager as fm
import platform
import re

# 忽略警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 核心密度参数 (保持原样)
# ==============================================================================
ACCURACY = 40           # 网格大小 (米)
DENSITY_THRESHOLD = 0.5 # 筛选高密度区域比例

# ------------------------------------------------------------------------------
# 基础工具函数 (逻辑保持不变)
# ------------------------------------------------------------------------------
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

def create_grid_params(df, accuracy):
    if df.empty: return None
    min_lng, max_lng = df['lon'].min(), df['lon'].max()
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    if min_lng == max_lng or min_lat == max_lat: return None
    accuracy_deg = accuracy / 111320
    return {'slon': min_lng, 'slat': min_lat, 'deltalon': accuracy_deg, 'deltalat': accuracy_deg}

# ------------------------------------------------------------------------------
# 单文件分析流程 (核心逻辑与命名完全保留)
# ------------------------------------------------------------------------------
def process_single_density_analysis(file_path, input_root, output_root):
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    # --- 【修改处 1】提取简洁的 Cluster 标签 (如 Cluster1) ---
    # 使用正则表达式提取 Cluster 及其数字，忽略大小写，不被 Trajectory 或 Cleaned 干扰
    match = re.search(r'(Cluster\d+)', file_base, re.IGNORECASE)
    cluster_label = match.group(1) if match else "Cluster"

    # 确定输出目录 (保持不变)
    rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
    curr_output_dir = os.path.join(output_root, rel_path)
    os.makedirs(curr_output_dir, exist_ok=True)
    
    print(f"\n📊 [密度分析] 处理: {file_name} -> {cluster_label}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    # 纠偏与命名逻辑 (保持不变)
    df = df.rename(columns={'经度': 'lon', '纬度': 'lat', 'lng': 'lon', 'latitude': 'lat', 'longitude': 'lon'})
    if 'lon' not in df.columns or 'lat' not in df.columns: return
    if df['lat'].abs().max() > 90 and df['lon'].abs().max() <= 90:
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})

    # 网格化计算 (保持不变)
    params = create_grid_params(df, ACCURACY)
    if not params: return
    df['LONCOL'], df['LATCOL'] = tbd.GPS_to_grid(df['lon'], df['lat'], params)
    
    grid_density = df.groupby(['LONCOL', 'LATCOL']).size().reset_index(name='count')
    if grid_density.empty: return

    # 密度筛选逻辑 (保持不变)
    limit_val = grid_density['count'].quantile(DENSITY_THRESHOLD)
    high_density_grids = grid_density[grid_density['count'] >= limit_val]
    top_traj = df.merge(high_density_grids[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how='inner')
    
    # --- 【修改处 2】保存结果 (简化文件名) ---
    csv_save_name = f"{cluster_label}_Spatial_Distribution.csv"
    top_traj.to_csv(os.path.join(curr_output_dir, csv_save_name), index=False, encoding='utf-8-sig')

    # 绘图逻辑 (保持不变)
    fix_chinese_font()
    grid_density['geometry'] = tbd.grid_to_polygon([grid_density['LONCOL'], grid_density['LATCOL']], params)
    grid_gdf = gpd.GeoDataFrame(grid_density, geometry='geometry')

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    grid_gdf.plot(
        ax=ax, column='count', cmap='Blues', scheme='quantiles',
        k=7, legend=True, edgecolor='none', alpha=0.9,
        legend_kwds={'loc': 'lower right', 'fmt': '{:.0f}'}
    )
    # 标题也同步简化
    ax.set_title(f"{cluster_label} Spatial Distribution ({ACCURACY}m Grid)", fontsize=14)
    ax.axis('off')
    
    # 图表保存命名简化
    img_save_name = f"{cluster_label}_Spatial_Distribution_Map.png"
    plt.savefig(os.path.join(curr_output_dir, img_save_name), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return True
# ------------------------------------------------------------------------------
# 🚀 封装的调用接口
# ------------------------------------------------------------------------------
def run_step_4_grid_density(city_name):

    INPUT_ROOT = os.path.join('data', 'Cleaning_data_after_segmentation', city_name)
    OUTPUT_ROOT = os.path.join('data', 'Density_analysis_results', city_name)
    
    print(f"\n[Step 4] 空间密度分析启动 | city: {city_name}")
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 找不到输入目录: {INPUT_ROOT}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    count = 0
    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            # 2. 修改匹配逻辑：增加对 "Cleaned" 或 "Trajectory" 的识别
            if file.endswith(".csv") and ("Cluster" in file or "cluster" in file.lower()) and ("Cleaned" in file or "Trajectory" in file):
                if process_single_density_analysis(os.path.join(root, file), INPUT_ROOT, OUTPUT_ROOT):
                    count += 1
    
    print(f"🎉 {city_name} 密度分析全部完成，共处理 {count} 个文件。")


