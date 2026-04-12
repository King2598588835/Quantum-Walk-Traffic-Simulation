# -*- coding: utf-8 -*-
"""
Step 1: Radius of Gyration Calculation and Clustering (模块化调用版)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持 (保持原样)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'FangSong', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 1. 定义计算回转半径 Rg 的函数 (逻辑完全保持不变)
# -------------------------------
def calculate_rg(df, lat_col='lat', lon_col='lon'):
    if len(df) < 2:
        return np.nan
    
    valid_df = df[
        (df[lat_col] >= -90) & (df[lat_col] <= 90) & 
        (df[lon_col] >= -180) & (df[lon_col] <= 180)
    ]
    
    if len(valid_df) < 2:
        return np.nan

    centroid_lat = valid_df[lat_col].mean()
    centroid_lon = valid_df[lon_col].mean()
    centroid = (centroid_lat, centroid_lon)
    
    distances_sq = []
    for _, row in valid_df.iterrows():
        try:
            d = geodesic((row[lat_col], row[lon_col]), centroid).meters
            distances_sq.append(d ** 2)
        except ValueError:
            continue
            
    if not distances_sq:
        return np.nan
        
    return np.sqrt(sum(distances_sq) / len(distances_sq))

# -------------------------------
# 2. 处理单个文件的函数 (逻辑完全保持不变)
# -------------------------------
def process_file_for_rg_clustering(file_path, output_folder, k=3):
    file_name = os.path.basename(file_path)
    file_basename = os.path.splitext(file_name)[0]
    file_output_folder = os.path.join(output_folder, file_basename)
    os.makedirs(file_output_folder, exist_ok=True)
    
    print(f"\n🚀 处理文件: {file_name}")
    
    try:
        # 1. 读取数据
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except:
            data = pd.read_csv(file_path, encoding='gbk')
        
        # 2. 检查并修复经纬度列
        required_cols = ['id', 'lat', 'lon']
        if not set(required_cols).issubset(data.columns):
            raise ValueError(f"❌ 缺少必要列: {required_cols}")

        max_lat = data['lat'].abs().max()
        max_lon = data['lon'].abs().max()
        if max_lat > 90 and max_lon <= 90:
            data = data.rename(columns={'lat': 'lon', 'lon': 'lat'})
        
        data = data[(data['lat'] >= -90) & (data['lat'] <= 90) & 
                    (data['lon'] >= -180) & (data['lon'] <= 180)]
        
        if data.empty:
            raise ValueError("❌ 所有数据均因坐标非法被剔除")

        # 3. 计算每条轨迹的Rg
        all_ids = data['id'].unique()
        rg_list = []
        for traj_id in tqdm(all_ids, desc="计算Rg", unit="traj"):
            traj = data[data['id'] == traj_id]
            rg = calculate_rg(traj)
            if not np.isnan(rg):
                rg_list.append({'id': traj_id, 'Rg': rg})
        
        rg_df = pd.DataFrame(rg_list)
        if rg_df.empty:
            raise ValueError("❌ 未能计算出有效的 Rg 值")

        rg_df.to_csv(os.path.join(file_output_folder, "回转半径汇总.csv"), index=False, encoding='utf-8-sig')
        
        # 4. 肘部法则 (绘图逻辑不变)
        X = rg_df[['Rg']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        k_range = range(1, 7)
        inertias = []
        for k_val in k_range:
            kmeans_temp = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, 'bo-', linewidth=2)
        plt.xlabel('聚类数量 k')
        plt.ylabel('簇内平方和 (Inertia)')
        plt.title(f'肘部法则 - {file_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(file_output_folder, "肘部法则分析图.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. 执行聚类 (K-means 逻辑不变)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        rg_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # 按照均值排序标签，确保分类的物理意义（小、中、大）
        cluster_mean_rg = rg_df.groupby('cluster')['Rg'].mean().sort_values()
        cluster_mapping = {old: new for new, old in enumerate(cluster_mean_rg.index)}
        rg_df['cluster_ordered'] = rg_df['cluster'].map(cluster_mapping)
        
        rg_df.to_csv(os.path.join(file_output_folder, "聚类标签结果.csv"), index=False, encoding='utf-8-sig')
        
        # 6. 拆分并保存轨迹数据 (中文文件名)
        data_with_cluster = data.merge(rg_df[['id', 'cluster_ordered']], on='id', how='left')
        data_with_cluster.rename(columns={'cluster_ordered': 'cluster'}, inplace=True)
        data_with_cluster.dropna(subset=['cluster'], inplace=True)
        data_with_cluster['cluster'] = data_with_cluster['cluster'].astype(int)
        
        for i in range(k):
            sub_data = data_with_cluster[data_with_cluster['cluster'] == i].drop(columns=['cluster'])
            save_p = os.path.join(file_output_folder, f"聚类{i+1}轨迹数据.csv")
            if not sub_data.empty:
                sub_data.to_csv(save_p, index=False, encoding='utf-8-sig')
        
        # 7. 绘制分布箱线图 (逻辑不变)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=rg_df, x='cluster_ordered', y='Rg', order=sorted(rg_df['cluster_ordered'].unique()))
        plt.yscale('log')
        plt.ylabel('回转半径 $R_g$ (米) - 对数坐标')
        plt.title(f'回转半径分布 - {file_name}')
        plt.savefig(os.path.join(file_output_folder, "回转半径分布图.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        return True

    except Exception as e:
        print(f"❌ 处理出错: {str(e)}")
        return False

# -------------------------------
# 🚀 封装的调用接口
# -------------------------------
def run_step_1_rg_clustering(city_name):
    """
    一键运行指定城市的回转半径计算与聚类
    """
    INPUT_FOLDER = os.path.join('data', '原始数据处理后', city_name)
    OUTPUT_FOLDER = os.path.join('data', '分类后数据', city_name)

    print(f"\n[Step 2] 回转半径分析启动 | 城市: {city_name}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    if not csv_files:
        print(f"❌ 在 {INPUT_FOLDER} 未找到CSV文件")
    else:
        for f in csv_files:
            process_file_for_rg_clustering(f, OUTPUT_FOLDER, k=3)
    
    print(f"🎉 {city_name} 回转半径聚类处理完成！")
