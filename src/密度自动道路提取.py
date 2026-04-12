# -*- coding: utf-8 -*-
"""
Step 4: Road Extraction Optimized (模块化调用版)
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
import matplotlib.pyplot as plt
import os
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from skimage import morphology, measure, filters, draw
from scipy.ndimage import gaussian_filter, convolve
from scipy.spatial import cKDTree
import warnings
import platform
import matplotlib.font_manager as fm
from tqdm import tqdm
import re

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 核心提取参数 (保持原样)
# ==============================================================================
ACCURACY = 40 
GAUSSIAN_SIGMA = 2.0
HIGH_PERCENTILE = 80
LOW_PERCENTILE  = 25 
CONNECT_THRESHOLD_DIST = 12 
SMOOTH_TOLERANCE = 0.0001

# ------------------------------------------------------------------------------
# 核心算法逻辑 (完全保持不变)
# ------------------------------------------------------------------------------
def fix_chinese_font():
    system_name = platform.system()
    font_path = None
    if system_name == "Windows":
        candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
        for path in candidates:
            if os.path.exists(path): font_path = path; break
    elif system_name == "Darwin":
        font_path = "/System/Library/Fonts/PingFang.ttc"
    if font_path:
        my_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = my_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False

def smart_connect_and_filter(binary_matrix, max_dist_px):
    matrix = binary_matrix.copy()
    labels = measure.label(matrix, connectivity=2)
    props = measure.regionprops(labels)
    if not props: return matrix
    props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
    main_prop = props_sorted[0]
    main_coords = main_prop.coords
    tree = cKDTree(main_coords)
    for prop in props_sorted[1:]:
        island_coords = prop.coords
        dists, idxs = tree.query(island_coords, k=1)
        if np.min(dists) <= max_dist_px:
            min_idx = np.argmin(dists)
            start_pt, end_pt = island_coords[min_idx], main_coords[idxs[min_idx]]
            rr, cc = draw.line(start_pt[0], start_pt[1], end_pt[0], end_pt[1])
            matrix[rr, cc] = 1
        else:
            matrix[labels == prop.label] = 0
    return matrix

def iterative_prune(skeleton, iterations):
    cleaned = skeleton.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    for _ in range(iterations):
        neighbors = convolve(cleaned, kernel, mode='constant', cval=0)
        is_endpoint = (neighbors >= 10) & (neighbors <= 11)
        if not np.any(is_endpoint): break
        cleaned[is_endpoint] = 0
    return cleaned.astype(bool)

def skeleton_to_lines(skeleton, params, simplify_tol):
    rows, cols = np.where(skeleton)
    def get_geo_coord(r, c):
        lon = params['slon'] + c * params['deltalon'] + params['deltalon']/2
        lat = params['slat'] + r * params['deltalat'] + params['deltalat']/2
        return (lon, lat)
    lines = []
    for r, c in zip(rows, cols):
        for nr, nc in [(r, c+1), (r+1, c), (r+1, c+1), (r+1, c-1)]:
            if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and skeleton[nr, nc]:
                lines.append(LineString([get_geo_coord(r, c), get_geo_coord(nr, nc)]))
    if not lines: return None
    merged = linemerge(lines)
    geoms = [merged] if isinstance(merged, LineString) else list(merged.geoms)
    smoothed = [line.simplify(simplify_tol, preserve_topology=True) for line in geoms]
    return gpd.GeoDataFrame(geometry=smoothed, crs="EPSG:4326")

def create_grid_params(df, accuracy):
    min_lng, max_lng, min_lat, max_lat = df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()
    accuracy_deg = accuracy / 111320
    buffer = accuracy_deg * 10
    return {
        'slon': min_lng - buffer, 'slat': min_lat - buffer,
        'deltalon': accuracy_deg, 'deltalat': accuracy_deg,
        'num_lng': int((max_lng - min_lng + 2*buffer) / accuracy_deg) + 2,
        'num_lat': int((max_lat - min_lat + 2*buffer) / accuracy_deg) + 2
    }

# ------------------------------------------------------------------------------
# 单文件分析流程 (逻辑零变动)
# ------------------------------------------------------------------------------
def process_single_road_extraction(input_file, input_root, output_root):
    file_name = os.path.basename(input_file)
    file_base = os.path.splitext(file_name)[0]
    
    if "聚类" in file_base: cluster_label = file_base.split("轨迹")[0]
    elif "cluster_" in file_base:
        match = re.search(r'cluster_(\d+)', file_base)
        cluster_label = f"聚类{int(match.group(1))+1}"
    else: cluster_label = "全量"

    rel_path = os.path.relpath(os.path.dirname(input_file), input_root)
    curr_output_dir = os.path.join(output_root, rel_path)
    os.makedirs(curr_output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_file)
        df = df.rename(columns={'经度': 'lon', '纬度': 'lat', 'lng': 'lon', 'latitude': 'lat', 'longitude': 'lon'})
        if 'lon' not in df.columns: return
    except: return

    params = create_grid_params(df, ACCURACY)
    df['LONCOL'], df['LATCOL'] = tbd.GPS_to_grid(df['lon'], df['lat'], params)
    grid_count = df.groupby(['LONCOL', 'LATCOL']).size().reset_index(name='count')
    heatmap_matrix = np.zeros((params['num_lat'], params['num_lng']), dtype=float)
    
    origin_lon_col, origin_lat_col = tbd.GPS_to_grid(params['slon'], params['slat'], params)
    rel_lon = (grid_count['LONCOL'] - origin_lon_col).astype(int)
    rel_lat = (grid_count['LATCOL'] - origin_lat_col).astype(int)
    valid = (rel_lon >= 0) & (rel_lon < params['num_lng']) & (rel_lat >= 0) & (rel_lat < params['num_lat'])
    heatmap_matrix[rel_lat[valid], rel_lon[valid]] = np.log1p(grid_count['count'][valid])
    
    smoothed = gaussian_filter(heatmap_matrix, sigma=GAUSSIAN_SIGMA)
    valid_vals = smoothed[smoothed > 0]
    if len(valid_vals) < 10: return
    binary = filters.apply_hysteresis_threshold(smoothed, np.percentile(valid_vals, LOW_PERCENTILE), np.percentile(valid_vals, HIGH_PERCENTILE))
    binary = morphology.binary_closing(binary, footprint=morphology.disk(2))
    connected_binary = smart_connect_and_filter(binary, max_dist_px=CONNECT_THRESHOLD_DIST)
    skeleton = morphology.skeletonize(connected_binary)
    final_skeleton = iterative_prune(skeleton, iterations=6)
    
    gdf_lines = skeleton_to_lines(final_skeleton, params, simplify_tol=SMOOTH_TOLERANCE)
    
    if gdf_lines is not None and not gdf_lines.empty:
        shp_save_name = f"{cluster_label}路网.shp"
        img_save_name = f"{cluster_label}路网预览图.png"
        gdf_lines.to_file(os.path.join(curr_output_dir, shp_save_name), encoding='utf-8')
        
        fix_chinese_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf_lines.plot(ax=ax, color='#2c3e50', linewidth=1.5)
        ax.set_title(f"{cluster_label} 道路提取结果")
        ax.axis('off')
        plt.savefig(os.path.join(curr_output_dir, img_save_name), dpi=200, bbox_inches='tight')
        plt.close()
        return True
    return False

# ------------------------------------------------------------------------------
# 🚀 封装的调用接口
# ------------------------------------------------------------------------------
def run_step_4_road_extraction(city_name):
    """
    一键运行指定城市的路网提取
    """
    INPUT_ROOT = os.path.join('data', '密度分析结果', city_name)
    OUTPUT_ROOT = os.path.join('data', '密度分析结果', city_name, "道路网络")

    print(f"\n[Step 4] 路网智能提取启动 | 城市: {city_name}")
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 找不到输入目录: {INPUT_ROOT}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    target_files = []
    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.endswith(".csv") and "轨迹空间分布" in file:
                target_files.append(os.path.join(root, file))
    
    if target_files:
        for f in tqdm(target_files, desc=f"Extracting {city_name}"):
            process_single_road_extraction(f, INPUT_ROOT, OUTPUT_ROOT)
        print(f"🎉 {city_name} 路网提取任务全部完成！")
    else:
        print(f"❌ {city_name} 下未找到轨迹分布文件。")

