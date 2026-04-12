# -*- coding: utf-8 -*-
"""
Step 4: Grid Density Analysis & Visualization (模块化调用版)
输入：Step 2 分类后的聚类数据
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
    
    # 提取聚类名称 (逻辑不变)
    if "聚类" in file_base:
        cluster_label = file_base.split("轨迹")[0]
    elif "cluster_" in file_base:
        match = re.search(r'cluster_(\d+)', file_base)
        cluster_num = int(match.group(1)) + 1 if match else 1
        cluster_label = f"聚类{cluster_num}"
    else:
        cluster_label = "全量"

    # 确定输出目录
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

    # 网格化计算
    params = create_grid_params(df, ACCURACY)
    if not params: return
    df['LONCOL'], df['LATCOL'] = tbd.GPS_to_grid(df['lon'], df['lat'], params)
    
    grid_density = df.groupby(['LONCOL', 'LATCOL']).size().reset_index(name='count')
    if grid_density.empty: return

    # 密度筛选逻辑
    limit_val = grid_density['count'].quantile(DENSITY_THRESHOLD)
    high_density_grids = grid_density[grid_density['count'] >= limit_val]
    top_traj = df.merge(high_density_grids[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how='inner')
    
    # 保存结果
    csv_save_name = f"{cluster_label}轨迹空间分布.csv"
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
    ax.set_title(f"{cluster_label}轨迹空间分布 ({ACCURACY}m网格)", fontsize=14)
    ax.axis('off')
    
    img_save_name = f"{cluster_label}轨迹空间分布图.png"
    plt.savefig(os.path.join(curr_output_dir, img_save_name), bbox_inches='tight', dpi=300)
    plt.close(fig)
    return True

# ------------------------------------------------------------------------------
# 🚀 封装的调用接口
# ------------------------------------------------------------------------------
def run_step_4_grid_density(city_name):
    """
    一键运行指定城市的网格密度分析
    """
    # 这里输入是 Step 2 的输出 (分类后的 CSV)
    INPUT_ROOT = os.path.join('data', '分类后数据', city_name)
    # 输出到指定的密度分析结果目录
    OUTPUT_ROOT = os.path.join('data', '密度分析结果', city_name)
    
    print(f"\n[Step 4] 空间密度分析启动 | 城市: {city_name}")
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 找不到输入目录: {INPUT_ROOT}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    count = 0
    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            # 匹配逻辑：包含“聚类”或“cluster_”的CSV文件，且排除已生成的分布文件
            if file.endswith(".csv") and ("聚类" in file or "cluster_" in file) and "空间分布" not in file:
                if process_single_density_analysis(os.path.join(root, file), INPUT_ROOT, OUTPUT_ROOT):
                    count += 1
    
    print(f"🎉 {city_name} 密度分析全部完成，共处理 {count} 个文件。")


