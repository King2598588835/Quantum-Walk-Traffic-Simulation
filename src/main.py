# -*- coding: utf-8 -*-
"""
Master Pipeline: 量子游走交通模拟全流程自动化调度
"""
import os
import sys

# --- 自动路径对齐逻辑 ---
# 获取当前脚本所在目录 (src/) 的父目录 (项目根目录)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
# 确保在根目录下运行，这样所有脚本内部的 'data/...' 相对路径都能生效
os.chdir(BASE_DIR)

# --- 导入各模块函数 ---
try:
    from Rg_Calculation_and_Classification import run_step_1_rg_clustering
    from Trajectory_processing import run_step_2_cleaning
    from MSD_Local_Diffusion_Index import run_step_3_msd_analysis
    from Density_zoning_calculation import run_step_4_grid_density
    from Build_road_graph import run_step_5_topology_building
    from Intelligent_roaming_quantum_fitting import run_quantum_solver
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保所有脚本文件都在 src/ 目录下，且文件名与 import 语句一致。")
    sys.exit(1)

def run_full_pipeline(city):
    """运行完整流水线"""
    try:
        print(f"\n{'='*50}")
        print(f"🚀 开始全流程处理: {city}")
        print(f"{'='*50}")
        
        # 1. 回转半径计算与聚类
        run_step_1_rg_clustering(city)
        # 2. 物理清洗与切分
        run_step_2_cleaning(city)
        # 3. 群体 MSD 与 Alpha 指数分析
        run_step_3_msd_analysis(city)
        # 4. 空间网格密度计算
        run_step_4_grid_density(city)
        # 5. 基于生成的空间密度构建路网

        # 6. 路网矢量拓扑构建
        run_step_5_topology_building(city)
        # 7. 量子游走前向拟合与参数回归
        run_quantum_solver(city)
        
        print(f"\n✨ {city} 全部流程处理成功！请在 data/ 目录下查看结果。")
    except Exception as e:
        print(f"\n❌ {city} 处理中断！")
        print(f"错误位置: {e}")

if __name__ == "__main__":
    # --- 待处理city列表 ---
    cities_to_process = ["Roman_data"] 
    
    for city in cities_to_process:
        run_full_pipeline(city)
        
    print("\n🎉 所有任务处理尝试结束。")