# -*- coding: utf-8 -*-
"""
Step 2 - 物理清洗与轨迹分割脚本 (模块化调用版)
"""

import pandas as pd
import numpy as np
import os
import glob
from geopy.distance import geodesic
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- 物理清洗参数 (保持原样) ---
AUTO_BBOX = True
MAX_SPEED_KMH = 150.0
MAX_TIME_GAP_MIN = 30.0
MAX_DURATION_HOURS = 24.0
MIN_POINTS = 5
USE_SMOOTHING = True
SMOOTH_WINDOW = 3

# ------------------------------------------------------------------------------
# 核心清洗工具函数 (逻辑完全保持不变)
# ------------------------------------------------------------------------------
def auto_standardize_columns(df):
    cols_lower = {str(c).lower(): c for c in df.columns}
    rename_map = {}
    if 'id' in cols_lower: rename_map[cols_lower['id']] = 'id'
    elif '车辆id' in df.columns: rename_map['车辆ID'] = 'id'
    if 'time' in cols_lower: rename_map[cols_lower['time']] = 'time'
    elif 'timestamp' in cols_lower: rename_map[cols_lower['timestamp']] = 'time'
    if 'lon' in cols_lower: rename_map[cols_lower['lon']] = 'lon'
    elif 'lng' in cols_lower: rename_map[cols_lower['lng']] = 'lon'
    if 'lat' in cols_lower: rename_map[cols_lower['lat']] = 'lat'
    if rename_map: df = df.rename(columns=rename_map)
    return df

def filter_bbox_logic(df):
    if df.empty: return df
    if AUTO_BBOX:
        min_lon, max_lon = df['lon'].quantile(0.001), df['lon'].quantile(0.999)
        min_lat, max_lat = df['lat'].quantile(0.001), df['lat'].quantile(0.999)
    else:
        min_lon, max_lon, min_lat, max_lat = 113.0, 115.0, 22.0, 23.0
    return df[(df['lon'] >= min_lon) & (df['lon'] <= max_lon) &
              (df['lat'] >= min_lat) & (df['lat'] <= max_lat)]

def clean_outliers_by_speed(df):
    df = df.sort_values('time').reset_index(drop=True)
    if len(df) < 2: return df
    keep_indices = [0]; last_valid_idx = 0
    lats, lons, times = df['lat'].values, df['lon'].values, df['time'].values
    for i in range(1, len(df)):
        try:
            if lats[i] == lats[last_valid_idx] and lons[i] == lons[last_valid_idx]: continue
            dist = geodesic((lats[last_valid_idx], lons[last_valid_idx]), (lats[i], lons[i])).meters
            dt = (times[i] - times[last_valid_idx]) / np.timedelta64(1, 's')
            if dt <= 0: continue
            speed_kmh = (dist / dt) * 3.6
            if speed_kmh <= MAX_SPEED_KMH:
                keep_indices.append(i); last_valid_idx = i
        except: continue
    return df.iloc[keep_indices].reset_index(drop=True)

def segment_trajectory(df):
    df = df.sort_values('time').reset_index(drop=True)
    df['dt'] = df['time'].diff().dt.total_seconds().fillna(0) / 60.0
    df['sub'] = (df['dt'] > MAX_TIME_GAP_MIN).astype(int).cumsum()
    df['new_id'] = df['id'].astype(str) + "_" + df['sub'].astype(str)
    return df.drop(columns=['dt', 'sub'])

def smooth_trajectory(df):
    if len(df) < SMOOTH_WINDOW: return df
    df = df.copy()
    df['lat'] = df['lat'].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    df['lon'] = df['lon'].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    return df

def process_single_file(file_path, input_root, output_root):
    filename = os.path.basename(file_path)
    file_base = os.path.splitext(filename)[0]
    
    if not filename.endswith('.csv'): return
    if any(k in filename for k in ['汇总', '结果', '分布', '标签']): return
    if '聚类' not in filename and 'cluster_' not in filename: return

    rel_path = os.path.relpath(file_path, input_root)
    rel_dir = os.path.dirname(rel_path)
    target_output_dir = os.path.join(output_root, rel_dir)
    if not os.path.exists(target_output_dir): os.makedirs(target_output_dir)

    try: 
        df = pd.read_csv(file_path, encoding='utf-8')
    except: 
        try: df = pd.read_csv(file_path, encoding='gbk')
        except: return
        
    df = auto_standardize_columns(df)
    if not {'lat', 'lon', 'time', 'id'}.issubset(df.columns): return

    if df['lat'].abs().max() > 90 and df['lon'].abs().max() <= 90:
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['lat','lon','time'], inplace=True)
    df = filter_bbox_logic(df) 
    if df.empty: return

    cleaned_segments = []
    for uid in df['id'].unique():
        traj = df[df['id'] == uid]
        if len(traj) < 2: continue
        traj = clean_outliers_by_speed(traj)
        if len(traj) < MIN_POINTS: continue
        segmented = segment_trajectory(traj)
        
        for new_uid, sub in segmented.groupby('new_id'):
            if len(sub) < MIN_POINTS: continue
            dur = (sub['time'].max() - sub['time'].min()).total_seconds() / 3600.0
            if dur > MAX_DURATION_HOURS: continue 
            if USE_SMOOTHING: sub = smooth_trajectory(sub)
            sub = sub.copy()
            sub['id'] = new_uid
            cleaned_segments.append(sub)
            
    if not cleaned_segments: return
    
    final_df = pd.concat(cleaned_segments)
    save_filename = f"{file_base.replace('轨迹数据', 'Trajectory_processing数据')}.csv" if "轨迹数据" in file_base else f"{file_base}_处理数据.csv"
    save_path = os.path.join(target_output_dir, save_filename)
    
    cols_to_keep = ['id', 'time', 'lat', 'lon']
    extra_cols = [c for c in final_df.columns if c not in cols_to_keep and c not in ['new_id', 'dt', 'sub']]
    final_df[cols_to_keep + extra_cols].to_csv(save_path, index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------------------
# 🚀 封装的调用接口
# ------------------------------------------------------------------------------
def run_step_2_cleaning(city_name):
    """
    只需调用此函数并传入city名，即可执行对应文件夹的清洗任务
    """
    INPUT_FOLDER = r'data\Classified_data\{}'.format(city_name)
    OUTPUT_FOLDER = r'data\Cleaning_data_after_segmentation\{}'.format(city_name)
    
    print(f"\n物理清洗任务启动 | city: {city_name}")
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 找不到输入目录: {INPUT_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    search_pattern = os.path.join(INPUT_FOLDER, "**", "*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    target_files = [f for f in files if ('聚类' in os.path.basename(f) or 'cluster_' in os.path.basename(f)) 
                    and "汇总" not in os.path.basename(f) 
                    and "标签" not in os.path.basename(f)]
    
    if target_files:
        for f in tqdm(target_files, desc=f"Processing {city_name}"): 
            process_single_file(f, INPUT_FOLDER, OUTPUT_FOLDER)
        print(f"🎉 {city_name} 清洗分割完成！")
    else: 
        print(f"❌ {city_name} 下未找到有效轨迹文件。")

