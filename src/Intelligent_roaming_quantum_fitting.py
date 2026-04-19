# -*- coding: utf-8 -*-
"""
Auto-Quantum Solver V4.1 - High Performance (Spectral Accelerated)
Optimized on: 2026-02-04
"""
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
import re
import scipy.linalg
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ 全局配置参数 (保持不变)
# ==============================================================================
CONFIG = {
    'WEIGHT_MSD': 3.0,     # MSD 拟合权重
    'WEIGHT_ALPHA': 1.0,   # Alpha 拟合权重
    'GAMMA_MAX': 2000.0,   # Gamma 搜索上限
    'SIM_STEPS': 100,      # 模拟步数
    'DT_SIM': 1.0,         # 模拟时间间隔
    'AUTO_SCALE': True,    # 自动对齐时间轴
    'FONTS': ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
}

plt.rcParams['font.sans-serif'] = CONFIG['FONTS']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 🛠️ 模块 1: 辅助工具函数
# ==============================================================================

def extract_cluster_id(filename):
    """从文件名提取 'Cluster_1' """
    match = re.search(r'(聚类\d+)', filename)
    return match.group(1) if match else None

def compute_alpha(msd, t, window_length=7, polyorder=2):
    """计算扩散指数 Alpha"""
    try:
        msd = np.maximum(msd, 1e-9)
        t = np.maximum(t, 1e-9)
        log_t = np.log(t)
        log_msd = np.log(msd)
        
        win = min(window_length, len(log_msd))
        if win % 2 == 0: win -= 1
        if win < 3: win = 3
        
        smooth_log_msd = savgol_filter(log_msd, window_length=win, polyorder=polyorder)
        alpha = np.gradient(smooth_log_msd, log_t)
        return alpha
    except:
        return np.zeros_like(msd)

def get_real_data_info(csv_path):
    """读取CSV数据"""
    try:
        df = pd.read_csv(csv_path)
        col_t = next((c for c in df.columns if 'Time' in c or 'time' in c), None)
        col_msd = next((c for c in df.columns if 'MSD' in c or 'msd' in c), None)
        
        if not col_t or not col_msd: return None
        
        t = df[col_t].values
        msd = df[col_msd].values
        mask = (t > 0) & (msd > 0) & (~np.isnan(msd))
        t_clean = t[mask]
        msd_clean = msd[mask]
        
        return {'time': t_clean, 'msd': msd_clean, 'alpha': compute_alpha(msd_clean, t_clean)}
    except:
        return None

# ==============================================================================
# 🕸️ 模块 2: 图构建与智能起点搜索
# ==============================================================================

def build_graph_for_cluster(cluster_id, topo_folder):
    """
    在指定的拓扑子文件夹中寻找 SHP
    """
    edge_pattern = os.path.join(topo_folder, f"*{cluster_id}*路网.shp")
    node_pattern = os.path.join(topo_folder, f"*{cluster_id}*路网节点.shp")
    
    edge_files = glob.glob(edge_pattern)
    node_files = glob.glob(node_pattern)
    
    if not edge_files or not node_files:
        return None, None, None, None

    try:
        nodes_gdf = gpd.read_file(node_files[0])
        edges_gdf = gpd.read_file(edge_files[0])
        G = nx.Graph()
        
        # 确定节点ID列
        node_key = 'node_id' if 'node_id' in nodes_gdf.columns else 'osmid'
        if node_key not in nodes_gdf.columns: 
            nodes_gdf['node_id'] = nodes_gdf.index
            node_key = 'node_id'

        for _, row in nodes_gdf.iterrows():
            G.add_node(row[node_key], pos=(row.geometry.x, row.geometry.y))
            
        for _, row in edges_gdf.iterrows():
            u = row['from_node'] if 'from_node' in row else row['u']
            v = row['to_node'] if 'to_node' in row else row['v']
            length = row['length'] if 'length' in row else (row['length_m'] if 'length_m' in row else 1.0)
            w = 1.0 / length if length > 0 else 1.0
            G.add_edge(u, v, weight=w)
            
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            nodes_gdf = nodes_gdf[nodes_gdf[node_key].isin(list(G.nodes))]
            
        return G, nodes_gdf, sorted(list(G.nodes())), node_key
    except:
        return None, None, None, None

def get_strategic_start_nodes(G, nodes_gdf, node_key):
    """Smart Scout: 寻找潜在起点"""
    node_list = list(G.nodes())
    if len(node_list) < 200: return node_list 

    candidates = set()
    # 1. 交通枢纽
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    candidates.update([n for n, d in degrees[:15]])
    # 2. 几何中心
    cx, cy = nodes_gdf.geometry.x.mean(), nodes_gdf.geometry.y.mean()
    d2 = (nodes_gdf.geometry.x - cx)**2 + (nodes_gdf.geometry.y - cy)**2
    candidates.add(nodes_gdf.loc[d2.idxmin(), node_key])
    # 3. 网格采样
    minx, miny, maxx, maxy = nodes_gdf.geometry.total_bounds
    xs = np.linspace(minx, maxx, 4); ys = np.linspace(miny, maxy, 4)
    for i in range(3):
        for j in range(3):
            tx, ty = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2
            d_grid = (nodes_gdf.geometry.x - tx)**2 + (nodes_gdf.geometry.y - ty)**2
            candidates.add(nodes_gdf.loc[d_grid.idxmin(), node_key])
            
    return list(set([n for n in candidates if n in G]))

# ==============================================================================
# 🧠 模块 3: 核心求解器 (⚡️ 已优化: 谱分解加速)
# ==============================================================================

def run_simulation_batch_fast(eig_vals, eig_vecs, start_idx, dist_sq, gammas):
    """
    使用谱分解(Spectral Decomposition)进行超高速模拟
    避免了循环中重复的矩阵指数运算 O(N^3) -> O(N^2)
    """
    # 初始态在特征基下的系数 c0 = V.T @ psi_0
    # 因为 psi_0 是独热向量(start_idx处为1)，所以 V.T @ psi_0 就是 V 的第 start_idx 行
    c_0 = eig_vecs[start_idx, :].astype(complex)
    
    results = {}
    n_steps = CONFIG['SIM_STEPS']
    dt = CONFIG['DT_SIM']
    
    # 预先分配内存
    seq = np.zeros(n_steps)
    
    for g in gammas:
        # 在特征空间计算演化相位： exp(-i * g * lambda * dt)
        # 这是一个向量运算，极快
        phase_step = np.exp(-1j * g * eig_vals * dt)
        
        # 当前系数
        c_curr = c_0.copy()
        
        # 时间步循环
        # 注意：这里我们逐个时间步更新，避免存储所有时间步的大矩阵
        seq_g = []
        for _ in range(n_steps):
            # 1. 在特征空间演化 (点乘)
            c_curr *= phase_step
            
            # 2. 投影回实空间: psi = V @ c
            psi = eig_vecs @ c_curr
            
            # 3. 计算概率和MSD
            prob = np.real(psi * np.conj(psi)) # 比 np.abs()**2 略快
            seq_g.append(np.dot(prob, dist_sq))
            
        results[g] = np.array(seq_g)
        
    return results

def fit_weighted(sim_results, real_t, real_msd, real_alpha):
    best_g, best_loss, best_pred = None, np.inf, None
    
    sim_steps = len(next(iter(sim_results.values())))
    sim_t = np.arange(1, sim_steps + 1) * CONFIG['DT_SIM']
    scale = real_t[0] / sim_t[0] if CONFIG['AUTO_SCALE'] else 1.0
    sim_t_scaled = sim_t * scale
    
    t_max = min(real_t.max(), sim_t_scaled.max())
    mask = real_t <= t_max
    if np.sum(mask) < 4: return None, np.inf, None
    
    tgt_t = real_t[mask]; tgt_msd = real_msd[mask]; tgt_alpha = real_alpha[mask]
    log_tgt_msd = np.log(tgt_msd + 1e-9)
    
    # 预计算插值所需的参数，加速循环
    # (此处依然使用遍历，但由于sim_results生成极快，整体耗时主要在此，逻辑保留)
    for g, y_sim in sim_results.items():
        f = interp1d(sim_t_scaled, y_sim, kind='linear', fill_value='extrapolate')
        y_sim_interp = f(tgt_t)
        
        # 线性回归求解比例系数 k (y_fit = k * y_sim)
        # nnls 求解 min ||Ax - b||
        res = nnls(y_sim_interp.reshape(-1, 1), tgt_msd)
        k = res[0][0]
        y_fit = y_sim_interp * k
        
        log_y_fit = np.log(np.maximum(y_fit, 1e-9))
        mse_msd = np.mean((log_tgt_msd - log_y_fit)**2)
        
        # 计算拟合的Alpha
        # 为了速度，只在loss有望更低时做精细计算，或者简化梯度计算
        # 这里保持原逻辑以确保精度
        alpha_fit = np.gradient(log_y_fit, np.log(tgt_t))
        valid = np.isfinite(alpha_fit) & np.isfinite(tgt_alpha)
        mse_alpha = np.mean((tgt_alpha[valid] - alpha_fit[valid])**2) if np.sum(valid) > 0 else 10.0
        
        loss = (CONFIG['WEIGHT_MSD'] * mse_msd) + (CONFIG['WEIGHT_ALPHA'] * mse_alpha)
        
        if loss < best_loss:
            best_loss = loss; best_g = g
            f_full = interp1d(sim_t_scaled, y_sim, kind='linear', fill_value='extrapolate')
            best_pred = f_full(real_t) * k
            
    return best_g, best_loss, best_pred

# ==============================================================================
# 💾 模块 4: 保存与绘图
# ==============================================================================

def save_single_result(output_dir, time, y_real, a_real, y_fit, a_fit, gamma, start_node, cluster_name):
    # 1. 绘图
    r2 = r2_score(y_real, y_fit)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    l1, = ax1.plot(time, y_real, 'o', ms=4, alpha=0.8, label='真实 MSD')
    l2, = ax1.plot(time, y_fit, '--', lw=2, label=f'拟合 MSD ($\gamma={gamma:.2f}$)')
    l3, = ax1.plot([], [], ' ', label=f'Start: {start_node}')
    l4, = ax1.plot([], [], ' ', label=f'$R^2$: {r2:.4f}')
    ax1.set_xlabel('Time'); ax1.set_ylabel('MSD'); ax1.legend(handles=[l1, l2, l3, l4]); ax1.grid(True, alpha=0.3)

    ax2.plot(time, a_real, '-o', ms=3, label='真实 Alpha')
    ax2.plot(time, a_fit, '-s', ms=3, label='拟合 Alpha')
    ax2.axhline(1.0, c='k', ls='--', alpha=0.5)
    ax2.fill_between(time, 0, 0.5, color='red', alpha=0.1)
    ax2.fill_between(time, 0.5, 1.0, color='yellow', alpha=0.1)
    ax2.fill_between(time, 1.0, 2.0, color='lightgreen', alpha=0.1)
    ax2.set_title('扩散指数 Alpha'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 3.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cluster_name}量子MSD变化以及扩散指数α图.png"), dpi=300)
    plt.close()
    
    # 2. CSV
    pd.DataFrame({
        'Time': time, 'Real_MSD': y_real, 'Real_Alpha': a_real, 'Fit_MSD': y_fit, 'Fit_Alpha': a_fit
    }).to_csv(os.path.join(output_dir, f"{cluster_name}量子MSD变化以及扩散指数α图.csv"), index=False)

# ==============================================================================
# 🎮 模块 5: 流程控制器 (The Manager)
# ==============================================================================

def process_single_cluster(csv_file, topo_folder, output_folder):
    """处理单个聚类文件"""
    filename = os.path.basename(csv_file)
    cluster_id = extract_cluster_id(filename)
    if not cluster_id: return None
    
    # 1. 数据
    info = get_real_data_info(csv_file)
    if not info: return None
    
    # 2. 图
    G, nodes_gdf, node_list, node_key = build_graph_for_cluster(cluster_id, topo_folder)
    if G is None: return None
    
    # ⚡️ 优化关键点：一次性计算特征值和特征向量
    # 路网是无向图，邻接矩阵实对称，使用 eigh 速度更快且数值稳定
    adj = nx.to_numpy_array(G, nodelist=node_list, weight='weight')
    eig_vals, eig_vecs = scipy.linalg.eigh(adj) 
    
    # 3. 找起点
    candidates = get_strategic_start_nodes(G, nodes_gdf, node_key)
    dists_map = {}
    
    # 预计算距离平方
    for n in candidates:
        try:
            d = nx.single_source_dijkstra_path_length(G, n, weight='length')
            # 确保节点顺序一致
            d_arr = np.array([d.get(x, 0) for x in node_list])
            dists_map[n] = d_arr**2
        except: pass
        
    best_node, min_loss = None, np.inf
    coarse_gammas = np.arange(10, CONFIG['GAMMA_MAX'], 50)
    
    # 遍历候选起点
    for sn in candidates:
        if sn not in dists_map: continue
        
        # 使用加速版模拟函数，传入特征值/向量而非矩阵
        sn_idx = node_list.index(sn)
        res = run_simulation_batch_fast(eig_vals, eig_vecs, sn_idx, dists_map[sn], coarse_gammas)
        
        _, loss, _ = fit_weighted(res, info['time'], info['msd'], info['alpha'])
        if loss < min_loss: min_loss = loss; best_node = sn
            
    # 4. 精搜 (Fine Search)
    if best_node is None: return None # 保护机制

    t_idx = node_list.index(best_node)
    t_dist = dists_map[best_node]
    
    # 第一轮精搜
    res_f = run_simulation_batch_fast(eig_vals, eig_vecs, t_idx, t_dist, np.arange(10, CONFIG['GAMMA_MAX'], 50))
    g1, _, _ = fit_weighted(res_f, info['time'], info['msd'], info['alpha'])
    
    # 第二轮精搜
    g_min2, g_max2 = max(0, g1-60), min(CONFIG['GAMMA_MAX'], g1+60)
    res_f2 = run_simulation_batch_fast(eig_vals, eig_vecs, t_idx, t_dist, np.arange(g_min2, g_max2, 5))
    g2, _, _ = fit_weighted(res_f2, info['time'], info['msd'], info['alpha'])
    
    # 最终微调
    g_min3, g_max3 = max(0, g2-10), min(CONFIG['GAMMA_MAX'], g2+10)
    res_final = run_simulation_batch_fast(eig_vals, eig_vecs, t_idx, t_dist, np.arange(g_min3, g_max3, 0.05))
    g_final, _, y_final = fit_weighted(res_final, info['time'], info['msd'], info['alpha'])
    
    a_final = compute_alpha(y_final, info['time'])
    save_single_result(output_folder, info['time'], info['msd'], info['alpha'], 
                       y_final, a_final, g_final, best_node, cluster_id)
    return cluster_id

def run_quantum_solver(dataset_name):
    """
    🔥 核心主入口函数 🔥
    参数: dataset_name (str) -> 例如 "Roman_data"
    """
    print(f"\n{'='*60}")
    print(f"🚀 启动量子求解器 (加速版) | 数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 1. 构建一级路径
    base_real = os.path.join(r'data', 'Group_MSD_analysis_results', dataset_name)
    base_topo = os.path.join(r'data', 'Construction_of_road_network_structure_topology', dataset_name)
    base_out  = os.path.join(r'data',  'forward_regression_result_quantum_walk', dataset_name)
    
    if not os.path.exists(base_real):
        print(f"❌ 错误：找不到数据目录 {base_real}")
        return

    # 2. 遍历底下的轨迹文件夹
    subfolders = [d for d in os.listdir(base_real) if os.path.isdir(os.path.join(base_real, d))]
    
    if not subfolders:
        print("⚠️ 未发现任何轨迹子文件夹。")
        return

    for folder_name in subfolders:
        print(f"\n📂 正在扫描轨迹集: {folder_name}")
        
        # 路径组装
        current_real_dir = os.path.join(base_real, folder_name)
        current_topo_dir = os.path.join(base_topo, folder_name)
        current_out_dir  = os.path.join(base_out, folder_name)
        
        # 检查路网是否存在
        if not os.path.exists(current_topo_dir):
            print(f"   ⚠️ 跳过：未在路网目录中找到对应的 {folder_name}")
            continue
            
        # 创建输出目录
        os.makedirs(current_out_dir, exist_ok=True)
        
        # 3. 遍历该文件夹下的 CSV (聚类文件)
        csv_files = glob.glob(os.path.join(current_real_dir, "*.csv"))
        if not csv_files:
            print("   ⚠️ 该文件夹下没有CSV文件。")
            continue
            
        print(f"   📊 发现 {len(csv_files)} 个聚类文件，开始处理...")
        
        # 进度条处理
        success_count = 0
        pbar = tqdm(csv_files, desc="   Processing", unit="file")
        for f in pbar:
            # 排除已生成的结果（防止重复处理）
            if "量子MSD" in f: continue
            
            try:
                cid = process_single_cluster(f, current_topo_dir, current_out_dir)
                if cid: 
                    success_count += 1
                    pbar.set_postfix({"Last": cid})
            except Exception as e:
                # 打印详细错误信息有助于调试
                # import traceback
                # traceback.print_exc()
                print(f"\n   ❌ 处理 {os.path.basename(f)} 时出错: {e}")
                
        print(f"   ✅ 完成！成功处理 {success_count} 个聚类。")