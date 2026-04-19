# -*- coding: utf-8 -*-
"""
Step 5: Road Network Topology Builder (模块化调用版)
核心：生成路网的节点(Nodes)与边(Edges)
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely import make_valid
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
import os
import warnings
import platform
import matplotlib.font_manager as fm
from tqdm import tqdm
import re

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 核心算法逻辑 (保持不变)
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

def get_utm_crs_from_gdf(gdf):
    try:
        if gdf.empty or gdf.geometry.isnull().all(): return None
        temp_gdf = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
        centroid = temp_gdf.geometry.unary_union.centroid
        lon, lat = centroid.x, centroid.y
        utm_band = int((lon + 180) / 6) + 1
        return f"EPSG:326{utm_band:02d}" if lat >= 0 else f"EPSG:327{utm_band:02d}"
    except: return "EPSG:3857"

def process_single_topology(file_path, input_root, output_root):
    filename = os.path.basename(file_path)
    file_base = os.path.splitext(filename)[0]
    
    if "Cluster" in file_base: cluster_label = file_base.split("路网")[0]
    elif "cluster_" in file_base:
        match = re.search(r'cluster_(\d+)', file_base)
        cluster_label = f"Cluster{int(match.group(1))+1}"
    else: cluster_label = "全量"

    rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
    save_dir = os.path.join(output_root, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    try:
        roads = gpd.read_file(file_path)
    except: return

    if len(roads) == 0: return
    roads = roads.explode(index_parts=False).reset_index(drop=True)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    roads['geometry'] = roads['geometry'].apply(make_valid)
    roads = roads[roads.geometry.type == 'LineString']
    
    if roads.crs is None: roads.crs = "EPSG:4326"
    target_crs = get_utm_crs_from_gdf(roads) if roads.crs.is_geographic else roads.crs
    roads = roads.to_crs(target_crs)
    
    # 拓扑融合逻辑
    merged_geometry = unary_union(roads.geometry)
    segments = [g for g in merged_geometry.geoms if isinstance(g, LineString)] if hasattr(merged_geometry, 'geoms') else ([merged_geometry] if isinstance(merged_geometry, LineString) else [])

    # 动态控制最大边长
    total_bounds = roads.total_bounds
    area_km2 = ((total_bounds[2] - total_bounds[0]) * (total_bounds[3] - total_bounds[1])) / 1e6
    max_edge_length = 300 if area_km2 < 1 else (800 if area_km2 < 10 else (2000 if area_km2 < 100 else 5000))
    
    refined_segments = []
    for seg in segments:
        length = seg.length
        if length <= max_edge_length: refined_segments.append(seg)
        else:
            num_sub = int(np.ceil(length / max_edge_length))
            sub_len = length / num_sub
            for i in range(num_sub):
                line = LineString([seg.interpolate(i * sub_len), seg.interpolate((i + 1) * sub_len)])
                if not line.is_empty and line.length > 0: refined_segments.append(line)
    
    # 节点提取与 ID 映射
    nodes = []
    for line in refined_segments: nodes.extend([Point(line.coords[0]), Point(line.coords[-1])])
    unique_nodes, seen_coords = [], set()
    for node in nodes:
        coord = (round(node.x, 4), round(node.y, 4))
        if coord not in seen_coords: seen_coords.add(coord); unique_nodes.append(node)
            
    nodes_count = len(unique_nodes)
    nodes_gdf = gpd.GeoDataFrame({'node_id': range(nodes_count)}, geometry=unique_nodes, crs=target_crs)
    coord_to_id = {(round(p.x, 4), round(p.y, 4)): i for i, p in enumerate(unique_nodes)}
    
    edges_data = []
    for line in refined_segments:
        start_id = coord_to_id.get((round(line.coords[0][0], 4), round(line.coords[0][1], 4)))
        end_id = coord_to_id.get((round(line.coords[-1][0], 4), round(line.coords[-1][1], 4)))
        if start_id is not None and end_id is not None:
            edges_data.append({'from_node': start_id, 'to_node': end_id, 'length_m': line.length, 'geometry': line})
    edges_gdf = gpd.GeoDataFrame(edges_data, crs=target_crs)
    
    # 保存结果
    edges_gdf.to_file(os.path.join(save_dir, f"{cluster_label}路网.shp"), encoding='utf-8')
    nodes_gdf.to_file(os.path.join(save_dir, f"{cluster_label}路网节点.shp"), encoding='utf-8')
    
    # 可视化预览
    fix_chinese_font()
    plt.figure(figsize=(10, 10))
    edges_gdf.plot(ax=plt.gca(), color='blue', linewidth=0.5, alpha=0.7)
    nodes_gdf.plot(ax=plt.gca(), color='red', markersize=2)
    plt.title(f'Topology: {cluster_label}')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{cluster_label}路网预览图.png"), dpi=200, bbox_inches='tight')
    plt.close()
    return True

# ------------------------------------------------------------------------------
# 🚀 封装的调用接口
# ------------------------------------------------------------------------------
def run_step_5_topology_building(city_name):
    """
    一键构建指定city路网的拓扑结构
    """
    # 这里输入是 Step 4 的输出结果
    INPUT_ROOT = os.path.join('data', 'Density_analysis_results', city_name, "road_network") 
    # 这里输出是 Step 5 的拓扑文件
    OUTPUT_ROOT = os.path.join('data', 'Construction_of_road_network_structure_topology', city_name) 
    
    print(f"\n[Step 5] 路网拓扑构建启动 | city: {city_name}")
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ 找不到输入目录: {INPUT_ROOT}")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    shp_files = []
    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith(".shp") and "节点" not in file and "node" not in file.lower() and "预览" not in file.lower():
                shp_files.append(os.path.join(root, file))
    
    if shp_files:
        for shp in tqdm(shp_files, desc=f"Building {city_name}"):
            process_single_topology(shp, INPUT_ROOT, OUTPUT_ROOT)
        print(f"🎉 {city_name} 拓扑构建完成！")
    else:
        print(f"❌ {city_name} 下未找到路网矢量文件。")

