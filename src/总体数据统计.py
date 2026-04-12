# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 2026
Description: 全自动适配版清洗脚本（自动识别城市范围 + 自动识别列名）
Input: 原始数据处理后\杭州市网约车数据
Output: 总体数据统计\杭州市网约车数据
"""
import pandas as pd
import numpy as np
import os
import glob
from geopy.distance import geodesic
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 🔧 参数配置区
# ==============================================================================
# 路径配置
INPUT_FOLDER = r'data\原始数据处理后\罗马数据'
OUTPUT_FOLDER = r'data\总体数据统计\罗马数据'

# 1. 空间范围过滤 (改为自动模式)
# True: 自动根据数据分布计算边界，剔除极个别离群点 (适用于任何城市)
# False: 不过滤空间范围 (不建议，因为会有 (0,0) 坐标噪点)
AUTO_BBOX = True 

# 2. 异常值过滤
MAX_SPEED_KMH = 150.0  # 最大速度 (km/h)

# 3. 轨迹分割 (停留检测)
MAX_TIME_GAP_MIN = 30.0  # 最大时间间隔 (分钟)，超过则切分为新轨迹

# 4. 时长过滤 (剔除异常长的轨迹)
MAX_DURATION_HOURS = 3

# 5. 轨迹平滑
USE_SMOOTHING = True
SMOOTH_WINDOW = 3

# 6. 过滤短轨迹
MIN_POINTS = 5
# ==============================================================================

def auto_standardize_columns(df):
    """
    自动标准化列名，兼容各种中文/英文格式
    """
    cols = df.columns.tolist()
    # 转小写比较
    cols_lower = {c.lower(): c for c in cols}
    
    rename_map = {}
    
    # 1. 识别 ID
    if 'id' in cols_lower: rename_map[cols_lower['id']] = 'id'
    elif '车辆id' in cols: rename_map['车辆ID'] = 'id'
    elif '车牌号' in cols: rename_map['车牌号'] = 'id'
    
    # 2. 识别 Time
    if 'time' in cols_lower: rename_map[cols_lower['time']] = 'time'
    elif '数据发送时间' in cols: rename_map['数据发送时间'] = 'time'
    elif '时间' in cols: rename_map['时间'] = 'time'
    
    # 3. 识别 Lon
    if 'lon' in cols_lower: rename_map[cols_lower['lon']] = 'lon'
    elif 'lng' in cols_lower: rename_map[cols_lower['lng']] = 'lon'
    elif '经度' in cols: rename_map['经度'] = 'lon'
    
    # 4. 识别 Lat
    if 'lat' in cols_lower: rename_map[cols_lower['lat']] = 'lat'
    elif '纬度' in cols: rename_map['纬度'] = 'lat'

    if rename_map:
        df = df.rename(columns=rename_map)
        
    return df

def filter_by_auto_bbox(df):
    """
    自动计算边界：保留经纬度在 0.1% 到 99.9% 分位数之间的数据
    这能有效去除 (0,0) 噪点，同时自动适应任何城市
    """
    if df.empty: return df
    
    # 计算分位数边界
    min_lon, max_lon = df['lon'].quantile(0.001), df['lon'].quantile(0.999)
    min_lat, max_lat = df['lat'].quantile(0.001), df['lat'].quantile(0.999)
    
    # 打印自动识别到的范围 (方便你确认是不是在正确的城市)
    print(f"  🌍 自动锁定城市范围: Lon[{min_lon:.3f}, {max_lon:.3f}], Lat[{min_lat:.3f}, {max_lat:.3f}]")
    
    original_len = len(df)
    df = df[
        (df['lon'] >= min_lon) & (df['lon'] <= max_lon) &
        (df['lat'] >= min_lat) & (df['lat'] <= max_lat)
    ]
    filtered_count = original_len - len(df)
    if filtered_count > 0:
        print(f"  ✂️  已自动剔除 {filtered_count} 个空间异常噪点")
        
    return df

def clean_outliers_by_speed(df, speed_limit_kmh):
    """速度去噪"""
    if len(df) < 2: return df
    
    # 确保按时间排序
    df = df.sort_values('time').reset_index(drop=True)
    
    lats = df['lat'].values
    lons = df['lon'].values
    times = df['time'].values
    
    # 向量化计算可能会更快，这里为了稳健保持循环（或优化为简单逻辑）
    # 这里使用简单的基于前一点的速度过滤
    keep_mask = np.ones(len(df), dtype=bool)
    last_valid_idx = 0
    
    for i in range(1, len(df)):
        # 快速估算距离 (近似) 或使用 geopy
        # 为提高效率，先跳过时间完全一样的点
        time_diff_s = (times[i] - times[last_valid_idx]) / np.timedelta64(1, 's')
        if time_diff_s <= 0:
            keep_mask[i] = False
            continue
            
        curr_pt = (lats[i], lons[i])
        last_pt = (lats[last_valid_idx], lons[last_valid_idx])
        
        try:
            dist_m = geodesic(last_pt, curr_pt).meters
        except:
            keep_mask[i] = False
            continue
            
        speed = (dist_m / time_diff_s) * 3.6
        
        if speed > speed_limit_kmh:
            keep_mask[i] = False # 标记为删除
        else:
            last_valid_idx = i # 更新有效点
            
    return df[keep_mask].reset_index(drop=True)

def segment_trajectory_by_time(df, time_gap_min):
    """时间分割"""
    df = df.sort_values('time').reset_index(drop=True)
    df['time_diff'] = df['time'].diff().dt.total_seconds() / 60.0
    df['time_diff'] = df['time_diff'].fillna(0)
    
    df['sub_id'] = (df['time_diff'] > time_gap_min).astype(int).cumsum()
    df['new_id'] = df['id'].astype(str) + "_" + df['sub_id'].astype(str)
    return df.drop(columns=['time_diff', 'sub_id'])

def smooth_trajectory(df, window=3):
    """轨迹平滑"""
    if len(df) < window: return df
    df = df.copy()
    df['lat'] = df['lat'].rolling(window=window, center=True, min_periods=1).mean()
    df['lon'] = df['lon'].rolling(window=window, center=True, min_periods=1).mean()
    return df

def process_single_file(file_path, output_path):
    filename = os.path.basename(file_path)
    print(f"\n📄 正在处理: {filename}")
    
    # 1. 读取
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return

    # 2. 自动标准化列名 (兼容不同文件格式)
    df = auto_standardize_columns(df)
    
    required_cols = {'lat', 'lon', 'time', 'id'}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ 跳过: 无法识别必要列 (当前列: {list(df.columns)})")
        return

    # 3. 基础格式转换
    # 经纬度反转检查
    if df['lat'].abs().max() > 90 and df['lon'].abs().max() <= 90:
        print("⚠️ 检测到经纬度反转，自动修正...")
        df = df.rename(columns={'lat': 'lon', 'lon': 'lat'})
    
    # 转换时间
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.dropna(subset=['lat', 'lon', 'time'], inplace=True)

    # 4. 全局空间过滤 (Auto-BBox)
    if AUTO_BBOX:
        df = filter_by_auto_bbox(df)

    if df.empty:
        print("⚠️ 数据在空间过滤后为空")
        return

    final_segments = []
    unique_ids = df['id'].unique()
    
    stats = {'duration_filtered': 0, 'short_filtered': 0}
    
    # 5. 逐轨迹处理
    for uid in tqdm(unique_ids, desc="  清洗进度", unit="traj"):
        traj = df[df['id'] == uid]
        if len(traj) < MIN_POINTS: continue

        # 速度去噪
        traj = clean_outliers_by_speed(traj, MAX_SPEED_KMH)
        if len(traj) < MIN_POINTS: 
            stats['short_filtered'] += 1
            continue
        
        # 时间分割
        traj_segmented = segment_trajectory_by_time(traj, MAX_TIME_GAP_MIN)
        
        for new_uid, sub_traj in traj_segmented.groupby('new_id'):
            if len(sub_traj) < MIN_POINTS: continue

            # 时长过滤
            duration_hours = (sub_traj['time'].max() - sub_traj['time'].min()).total_seconds() / 3600.0
            if duration_hours > MAX_DURATION_HOURS:
                stats['duration_filtered'] += 1
                continue 
                
            # 平滑
            if USE_SMOOTHING:
                sub_traj = smooth_trajectory(sub_traj, window=SMOOTH_WINDOW)
            
            sub_traj = sub_traj.copy()
            sub_traj['id'] = new_uid
            final_segments.append(sub_traj)
                
    # 6. 保存
    if final_segments:
        result_df = pd.concat(final_segments)
        cols = ['id', 'time', 'lat', 'lon']
        # 尝试保留原有的速度/方向列，如果存在
        extra_cols = [c for c in result_df.columns if c not in cols and c != 'new_id']
        result_df = result_df[cols + extra_cols]
        
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 处理完成 -> 产生 {result_df['id'].nunique()} 条轨迹")
    else:
        print(f"⚠️ {filename} 处理后无有效轨迹。")

# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    
    if not csv_files:
        print(f"❌ 未找到文件: {INPUT_FOLDER}")
    else:
        print(f"🚀 开始全自动处理 {len(csv_files)} 个文件 (Auto-BBox Mode)")
        for f in csv_files:
            out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(f))
            process_single_file(f, out_path)
        print("\n🎉 全部完成！")