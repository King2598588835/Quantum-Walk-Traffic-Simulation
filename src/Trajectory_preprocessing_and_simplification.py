# -*- coding: utf-8 -*-
"""
科研trajectory_processing_data处理万能版
1. 支持Roman_data格式 (.txt, POINT坐标, mixed时间)
2. 支持国内常见格式 (杭州、厦门、车牌号等 CSV)
3. 包含轨迹预处理、去噪、轨迹简化 (DP算法)
"""
import transbigdata as tbd
import pandas as pd
from shapely.geometry import Point, LineString
from pyproj import Transformer
from geopy.distance import geodesic
from tqdm import tqdm
import warnings
import os
import glob
import numpy as np

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 核心解析与标准化模块
# =============================================================================

def smart_load_data(file_path):
    """
    智能读取函数：自动区分罗马TXT格式和普通CSV格式
    """
    ext = os.path.splitext(file_path)[-1].lower()
    
    # --- 逻辑 A: 处理罗马格式 (.txt 或包含特定特征) ---
    if ext == '.txt':
        try:
            print(f"  识别为文本格式，尝试按Roman_data规范解析...")
            # 罗马格式：分号分隔，无表头
            df = pd.read_csv(file_path, sep=';', header=None, names=['id', 'time', 'geo'])
            
            if 'geo' in df.columns and df['geo'].str.contains('POINT', na=False).any():
                print("  ✨ 确认Roman_data特征: 解析 POINT 坐标...")
                # 解析 POINT(41.88 12.48)
                temp_geo = df['geo'].str.replace('POINT(', '', regex=False).str.replace(')', '', regex=False)
                coords = temp_geo.str.split(' ', expand=True)
                df['lat'] = coords[0].astype(float)
                df['lon'] = coords[1].astype(float)
                # 时间格式 mixed
                df['time'] = pd.to_datetime(df['time'], format='mixed')
                return df[['id', 'time', 'lon', 'lat']], True
        except Exception as e:
            print(f"  ⚠️ TXT解析失败，尝试普通模式: {e}")

    # --- 逻辑 B: 处理普通 CSV 格式 ---
    encodings = ['utf-8', 'gbk', 'gb18030']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"  📖 读取成功 (编码: {enc})")
            return df, True
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            return None, False
    return None, False

def standardize_columns(df):
    """
    自动识别列名并统一为 [id, time, lon, lat]
    """
    columns = df.columns.tolist()
    
    # 如果已经是标准格式则直接返回
    if all(c in columns for c in ['id', 'time', 'lon', 'lat']):
        return df, True

    # 映射字典
    mapping = {
        'cid': 'id', '车辆ID': 'id', '车牌号': 'id', 'DEVICEID': 'id',
        '经度': 'lon', 'lng': 'lon', 'Longitude': 'lon',
        '纬度': 'lat', 'Latitude': 'lat',
        '数据发送时间': 'time', '数据接收时间': 'time', '时间': 'time', 'DateTime': 'time'
    }
    
    new_columns = {}
    for col in columns:
        if col in mapping:
            new_columns[col] = mapping[col]
    
    if 'lon' in new_columns.values() or 'id' in new_columns.values():
        df = df.rename(columns=new_columns)
        print(f"  💡 列名已标准化: {new_columns}")
        return df, True
    
    return df, False

# =============================================================================
# 2. Trajectory_processing核心函数 (保持逻辑不变)
# =============================================================================

def preprocess_trajectory(df, id_col='id', time_col='time', lat_col='lat', lon_col='lon',
                          disgap=5000, timegap=1800, speedlimit=120, dislimit=2000, anglelimit=30):
    print("  🔹 正在进行预处理 (重编号/去漂移/去冗余)...")
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df.dropna(subset=[lat_col, lon_col, time_col], inplace=True)
    
    # 步骤：空间跳跃编号 -> 时间间隔编号 -> 清除漂移 -> 冗余点
    temp_df = tbd.id_reindex_disgap(df, col=[id_col, lon_col, lat_col], disgap=disgap, suffix='_new')
    temp_df = tbd.id_reindex(temp_df, 'id_new', new=True, timegap=timegap, timecol=time_col, suffix=False)
    temp_df[id_col] = temp_df['id_new']
    
    try:
        temp_df = tbd.traj_clean_drift(temp_df, col=[id_col, time_col, lon_col, lat_col], 
                                     speedlimit=speedlimit, dislimit=dislimit, anglelimit=anglelimit)
    except: pass
    
    temp_df = tbd.traj_clean_redundant(temp_df, col=[id_col, time_col, lon_col, lat_col])
    temp_df.sort_values([id_col, time_col], inplace=True)
    return temp_df

def simplify_trajectory(df, W=20, R=30):
    """ DP算法轨迹简化 """
    df_work = df.copy()
    if df_work.empty: return df_work
    
    simplified_rows = []
    traj_ids = df_work['id'].unique()
    
    for traj_id in tqdm(traj_ids, desc="  📉 简化轨迹", leave=False):
        group = df_work[df_work['id'] == traj_id].reset_index(drop=True)
        if len(group) <= 2:
            simplified_rows.extend(group.to_dict('records'))
            continue
        
        # 投影转换
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        
        # --- 简化逻辑 ---
        key_indices = [0]
        stack = [(0, len(group)-1)]
        while stack:
            start, end = stack.pop()
            if start + 1 >= end: continue
            
            # 创建线段
            s_xy = transformer.transform(group.iloc[start]['lon'], group.iloc[start]['lat'])
            e_xy = transformer.transform(group.iloc[end]['lon'], group.iloc[end]['lat'])
            line = LineString([s_xy, e_xy])
            
            max_dist = 0
            idx_max = start
            for i in range(start + 1, end):
                p_xy = transformer.transform(group.iloc[i]['lon'], group.iloc[i]['lat'])
                d = Point(p_xy).distance(line)
                if d > max_dist:
                    max_dist = d
                    idx_max = i
            
            if max_dist > W:
                stack.append((start, idx_max))
                stack.append((idx_max, end))
                key_indices.append(idx_max)
        
        key_indices.append(len(group)-1)
        res = group.iloc[sorted(list(set(key_indices)))]
        
        # 距离 R 过滤
        filtered = []
        for _, row in res.iterrows():
            if not filtered: filtered.append(row.to_dict())
            else:
                prev = filtered[-1]
                if geodesic((prev['lat'], prev['lon']), (row['lat'], row['lon'])).meters >= R:
                    filtered.append(row.to_dict())
        simplified_rows.extend(filtered)
        
    return pd.DataFrame(simplified_rows)

# =============================================================================
# 3. 主循环处理逻辑
# =============================================================================

def process_file(input_path, output_path):
    print(f"\n🚀 正在处理: {os.path.basename(input_path)}")
    
    # 1. 智能读取
    data, success = smart_load_data(input_path)
    if not success: return False
    
    # 2. 列名标准化
    data, success = standardize_columns(data)
    if not success: 
        print("  ❌ 无法识别列名结构")
        return False
    
    # 3. 时间格式转换 (如果是 CSV 的话需要再次确保)
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'], errors='coerce', format='mixed')
    data.dropna(subset=['time'], inplace=True)
    
    # 4. 排序与预处理
    data.sort_values(['id', 'time'], inplace=True)
    data = preprocess_trajectory(data)
    
    # 5. 轨迹简化
    data = simplify_trajectory(data, W=20, R=30)
    
    # 6. 过滤过短轨迹 & ID重置
    id_counts = data['id'].value_counts()
    data = data[data['id'].isin(id_counts[id_counts >= 5].index)]
    
    unique_ids = data['id'].unique()
    id_map = {old: new for new, old in enumerate(unique_ids, 1)}
    data['id'] = data['id'].map(id_map)
    
    # 7. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data[['id', 'time', 'lon', 'lat']].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ 处理完成，保存至: {output_path}")
    return True

if __name__ == "__main__":
    # --- 配置区 ---
    # 这里可以放任何格式的文件 (Rome的txt, 厦门的csv, 杭州的csv)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, "data", "original_data","Roman_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "AfterProcessing", "Roman_data")

    files = glob.glob(os.path.join(INPUT_DIR, "*.*")) # 读取所有后缀文件
    print(f"📂 找到 {len(files)} 个文件")
    
    for f in files:
        out = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(f))[0] + "_cleaned.csv")
        process_file(f, out)