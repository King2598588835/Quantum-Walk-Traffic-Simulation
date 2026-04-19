import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 1. 设置中文字体（解决标题乱码问题）
# 自动寻找系统中的黑体，如果没有则使用默认
def set_ch_font():
    for font in ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False

set_ch_font()

# 2. 读取数据
# 请确保路径正确，Windows下建议使用反斜杠或原始字符串
edge_path = r"聚类2\network_edges.shp"
node_path = r"聚类2\network_nodes.shp"

edges = gpd.read_file(edge_path)
nodes = gpd.read_file(node_path)

# --- 图 1：道路提取结果 (仅路网) ---
fig, ax1 = plt.subplots(figsize=(10, 12), dpi=300)
fig.patch.set_facecolor('white')  # 设置背景为白色

# 绘制路网：颜色使用深灰蓝色，线宽适中
edges.plot(ax=ax1, color='#2c3e50', linewidth=1.5, zorder=1)

# 设置标题和显示效果
ax1.set_title('聚类1 道路提取结果', fontsize=18, pad=20)
ax1.set_axis_off()  # 隐藏坐标轴

# 调整布局并保存
plt.tight_layout()
plt.savefig('road_network_only.png', bbox_inches='tight', facecolor='white')
print("图1：道路提取结果已生成。")
plt.show()


# --- 图 2：Topology 拓扑结构 (路网 + 节点) ---
fig, ax2 = plt.subplots(figsize=(10, 12), dpi=300)
fig.patch.set_facecolor('white')

# 绘制路网：蓝色细线
edges.plot(ax=ax2, color='blue', linewidth=0.5, alpha=0.6, zorder=1)

# 绘制节点：红色小圆点
nodes.plot(ax=ax2, color='red', markersize=5, zorder=2)

# 设置标题 (注意：图片上显示的是 Topology: 聚类1)
ax2.set_title('Topology: 聚类1', fontsize=18, pad=20)
ax2.set_axis_off()

# 调整布局并保存
plt.tight_layout()
plt.savefig('topology_visualization.png', bbox_inches='tight', facecolor='white')
print("图2：拓扑结构图已生成。")
plt.show()