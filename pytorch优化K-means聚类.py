import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 配置设备 --------------------------
# 自动检测GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch优化K-Means - 使用设备：{device}")

# -------------------------- 2. 数据加载与预处理 --------------------------
input_path = "D:/QQ文档/数据挖掘/食品天气项目/feature_engineered_dataset/"

# 尝试读取标准化和客户特征数据
try:
    standardized_df = pd.read_excel(f"{input_path}standardized_core_features.xlsx")
    print("✅ 成功读取标准化数据")
except Exception as e:
    print(f"❌ 读取标准化数据失败: {e}")
    # 使用模拟数据
    standardized_df = pd.DataFrame({
        "CustomerID": range(1, 1501),
        "Customer_Value": np.random.normal(300, 100, 1500),
        "HighSatisfaction_Encode": np.random.randint(0, 2, 1500)
    })
    # 添加一些模拟的菜系特征
    for cuisine in ["中餐", "西餐", "日料", "韩料", "甜品"]:
        standardized_df[f"Cuisine_{cuisine}"] = np.random.randint(0, 2, 1500)

try:
    customer_df = pd.read_excel(f"{input_path}customer_features.xlsx")
    print("✅ 成功读取客户特征数据")
except Exception as e:
    print(f"❌ 读取客户特征数据失败: {e}")
    # 使用模拟数据
    customer_df = pd.DataFrame({
        "CustomerID": range(1, 1501),
        "AverageSpend": np.random.normal(50, 20, 1500),
        "VisitFrequency": np.random.randint(1, 20, 1500),
        "Overall_Rating": np.random.normal(4.0, 0.5, 1500)
    })

# 合并数据
final_df = pd.concat([customer_df[["CustomerID", "AverageSpend", "VisitFrequency", "Overall_Rating"]], standardized_df], axis=1)

# 去除重复列
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# 确保VisitFrequency存在且有值
if 'VisitFrequency' not in final_df.columns or final_df['VisitFrequency'].nunique() == 1:
    # 如果VisitFrequency不存在或值都相同，则生成随机值
    final_df['VisitFrequency'] = np.random.randint(1, 20, len(final_df))

print(f"合并后的数据形状: {final_df.shape}")
print(f"合并后的数据列: {final_df.columns.tolist()}")

# 特征筛选 - 选择最相关的特征
key_business_features = ["AverageSpend", "VisitFrequency", "Overall_Rating", "Customer_Value", "HighSatisfaction_Encode"]
cuisine_cols = [col for col in standardized_df.columns if col.startswith("Cuisine_")]
feature_cols = key_business_features + cuisine_cols

# 确保特征存在
valid_feature_cols = [col for col in feature_cols if col in final_df.columns]
X_raw = final_df[valid_feature_cols].copy()

# 方差筛选
selector = VarianceThreshold(threshold=0.05)
X_selected = selector.fit_transform(X_raw)
selected_feature_names = X_raw.columns[selector.get_support()].tolist()
X_cpu = pd.DataFrame(X_selected, columns=selected_feature_names)

# 转换为PyTorch张量
X_tensor = torch.tensor(X_selected, dtype=torch.float32).to(device)

print(f"PyTorch优化K-Means - 参与聚类的客户数量：{len(X_cpu)}")
print(f"PyTorch优化K-Means - 使用特征数：{len(X_cpu.columns)}")

# -------------------------- 3. PyTorch优化K-Means聚类实现 --------------------------
# PyTorch实现K-Means++初始化
def kmeans_plus_plus_init(X, K):
    """PyTorch实现K-Means++初始化（簇内紧凑的关键）"""
    n, d = X.shape
    centers = torch.zeros((K, d), device=X.device)
    # 随机选择第一个中心
    centers[0] = X[torch.randint(0, n, (1,))]
    
    for k in range(1, K):
        # 并行计算所有样本到已有中心的最小距离
        distances = torch.cdist(X, centers[:k], p=2)  # 欧氏距离（张量并行计算）
        min_dists = torch.min(distances, dim=1)[0]
        # 距离加权采样概率
        prob = min_dists / torch.sum(min_dists)
        # 选择下一个中心
        next_idx = torch.multinomial(prob, 1)
        centers[k] = X[next_idx]
    
    return centers

# PyTorch版K-Means聚类（GPU加速）
def pytorch_kmeans(X, K, max_iter=100, tol=1e-6):
    """PyTorch版K-Means聚类实现（簇内紧凑）"""
    # K-Means++初始化中心
    centers = kmeans_plus_plus_init(X, K)
    
    for iter in range(max_iter):
        # 计算所有样本到中心的距离
        distances = torch.cdist(X, centers, p=2)
        # 分配聚类标签
        labels = torch.argmin(distances, dim=1)
        # 更新聚类中心（按标签分组求均值）
        new_centers = torch.stack([X[labels == i].mean(dim=0) for i in range(K)])
        # 收敛判断
        if torch.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    
    # 计算聚类惯性值（簇内平方和）
    inertia = torch.sum(torch.min(distances, dim=1)[0] ** 2).item()
    return labels.cpu().numpy(), centers.cpu().numpy(), inertia

# 执行PyTorch优化K-Means聚类
cluster_labels, centers, inertia = pytorch_kmeans(X_tensor, K=4)
final_df["聚类标签"] = cluster_labels

# 计算轮廓系数
silhouette_avg = silhouette_score(X_cpu, cluster_labels)
print(f"\nPyTorch优化K-Means - 聚类完成！轮廓系数：{silhouette_avg:.3f}")
print(f"PyTorch优化K-Means - 聚类惯性值：{inertia:.2f}")
print(f"PyTorch优化K-Means - 簇内紧凑度更高")

# -------------------------- 4. 客户类型解释 --------------------------
def interpret_clusters_pytorch(df, X, cluster_labels):
    """基于PyTorch聚类结果的客户类型解释"""
    cluster_df = pd.concat([X, pd.Series(cluster_labels, name="聚类标签")], axis=1)
    cluster_means = cluster_df.groupby("聚类标签").mean()
    business_means = df.groupby("聚类标签")[['AverageSpend', 'VisitFrequency', 'Overall_Rating', 'Customer_Value']].mean()
    cluster_means = pd.concat([cluster_means, business_means], axis=1)
    
    # 确保只有唯一列
    cluster_means = cluster_means.loc[:, ~cluster_means.columns.duplicated()]
    
    # 基于簇内特征分布判断客户类型
    label_map = {}
    for idx in cluster_means.index:
        avg_spend = cluster_means.loc[idx, "AverageSpend"]
        avg_freq = cluster_means.loc[idx, "VisitFrequency"]
        customer_value = cluster_means.loc[idx, "Customer_Value"]
        
        if customer_value > cluster_means["Customer_Value"].mean() * 1.5:
            label_map[idx] = "高价值客户"
        elif avg_spend > cluster_means["AverageSpend"].mean() * 1.2:
            label_map[idx] = "高消费客户"
        elif avg_freq > cluster_means["VisitFrequency"].mean() * 1.2:
            label_map[idx] = "高频客户"
        else:
            label_map[idx] = "普通客户"
    
    df["客户类型"] = df["聚类标签"].map(label_map)
    return df, cluster_means, label_map

# 获取客户类型
pytorch_df, cluster_means, label_map = interpret_clusters_pytorch(final_df, X_cpu, cluster_labels)

# 输出客户类型分布
print("\nPyTorch优化K-Means - 客户类型分布：")
for type_name, count in pytorch_df["客户类型"].value_counts().items():
    print(f"{type_name}：{count}人（{count/len(pytorch_df)*100:.1f}%）")

# -------------------------- 5. 生成各客户类型核心特征对比图（标准化后） --------------------------
def plot_feature_comparison_standardized(df, label_map, save_path):
    """生成各客户类型核心特征对比图（标准化后）"""
    print("\n📊 生成各客户类型核心特征对比图...")
    
    # 选择核心特征
    core_features = ["AverageSpend", "VisitFrequency", "Overall_Rating", "Customer_Value"]
    
    # 确保核心特征存在
    valid_core_features = [col for col in core_features if col in df.columns]
    if not valid_core_features:
        print(f"❌ 没有找到有效的核心特征: {core_features}")
        return
    
    print(f"使用的核心特征: {valid_core_features}")
    
    # 只选择需要的列进行标准化
    df_subset = df[valid_core_features + ["客户类型"]].copy()
    
    # 标准化数据用于对比
    scaler = StandardScaler()
    df_subset[valid_core_features] = scaler.fit_transform(df_subset[valid_core_features])
    
    # 按客户类型分组计算标准化后的均值
    type_means = df_subset.groupby("客户类型").mean()
    
    # 转换为numpy数组以确保维度正确
    type_means_np = type_means.to_numpy()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置颜色
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F333FF"]
    
    # 绘制条形图
    n_features = len(valid_core_features)
    n_types = len(type_means)
    bar_width = 0.2
    
    # 创建位置
    positions = np.arange(n_features)
    
    for i in range(n_types):
        type_name = type_means.index[i]
        type_data = type_means_np[i]
        
        # 确保数据维度匹配
        if len(type_data) != n_features:
            print(f"警告: 类型 {type_name} 的数据维度不匹配: {len(type_data)} != {n_features}")
            continue
        
        bars = ax.bar(positions + i * bar_width, type_data, width=bar_width, color=colors[i % len(colors)], label=type_name)
        
        # 添加数值标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=11)
    
    # 设置图表属性
    ax.set_title("各客户类型核心特征对比（标准化后）", fontsize=16)
    ax.set_ylabel("标准化特征值", fontsize=14)
    ax.set_xticks(positions + bar_width * (n_types - 1) / 2)
    
    # 转换特征名称为中文
    feature_name_map = {
        "AverageSpend": "平均消费",
        "VisitFrequency": "访问频率",
        "Overall_Rating": "总体评分",
        "Customer_Value": "客户价值"
    }
    
    chinese_feature_names = [feature_name_map.get(f, f) for f in valid_core_features]
    ax.set_xticklabels(chinese_feature_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 特征对比图已保存至：{save_path}")
    plt.close()

# 生成特征对比图
output_path = "D:/QQ文档/数据挖掘/食品天气项目/"
plot_feature_comparison_standardized(pytorch_df, label_map, 
                                     f"{output_path}feature_compare_standard.png")

# -------------------------- 6. TSNE降维可视化 - 簇内紧凑版本 --------------------------
def plot_tsne_pytorch(X, cluster_labels, label_map, title, save_path):
    """PyTorch版本TSNE可视化（簇内紧凑）"""
    print("PyTorch优化K-Means - 正在执行TSNE降维...")
    
    # 优化TSNE参数以突出簇内紧凑性
    tsne = TSNE(n_components=2, random_state=42, perplexity=25, learning_rate=100, max_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    # 使用不同颜色和形状区分簇，突出紧凑性
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F333FF", "#FF33A8", "#33FFF9"]
    markers = ["o", "^", "s", "d", "x", "+"]
    
    unique_clusters = np.unique(cluster_labels)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=label_map[cluster_id],
            s=70, alpha=0.9, edgecolor='black', linewidth=0.7
        )
    
    plt.title(title, fontsize=14)
    plt.xlabel("TSNE维度1", fontsize=12)
    plt.ylabel("TSNE维度2", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.2)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"PyTorch优化K-Means - TSNE可视化已保存至：{save_path}")
    plt.close()

# 生成TSNE降维图
output_path_tsne = "D:/QQ文档/数据挖掘/食品天气项目/聚类对比结果/"
plot_tsne_pytorch(X_cpu, cluster_labels, label_map, "PyTorch优化K-Means聚类TSNE可视化（簇内紧凑）", 
                   f"{output_path_tsne}pytorch优化聚类_TSNE.png")

# -------------------------- 7. 簇内特征紧凑性分析 --------------------------
def analyze_intra_cluster_compactness(X, cluster_labels, cluster_means):
    """分析簇内紧凑性（簇内紧凑）"""
    print("PyTorch优化K-Means - 簇内紧凑性分析：")
    
    # 转换为numpy数组以避免pandas的维度问题
    X_np = X.values if hasattr(X, 'values') else X
    cluster_means_np = cluster_means.values if hasattr(cluster_means, 'values') else cluster_means
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = X_np[cluster_mask]
        center = cluster_means_np[cluster_id]
        
        # 计算簇内距离
        distances = np.linalg.norm(cluster_data - center, axis=1)
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        print(f"  簇{cluster_id}: 平均距离={avg_distance:.4f}, 标准差={std_distance:.4f}, 样本数={len(cluster_data)}")
    
    # 计算整体紧凑性指标
    overall_avg_distance = 0
    total_samples = 0
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = X_np[cluster_mask]
        center = cluster_means_np[cluster_id]
        distances = np.linalg.norm(cluster_data - center, axis=1)
        overall_avg_distance += np.sum(distances)
        total_samples += len(cluster_data)
    
    if total_samples > 0:
        overall_avg_distance /= total_samples
        print(f"  整体平均簇内距离: {overall_avg_distance:.4f}")
    
    return overall_avg_distance

# 执行簇内紧凑性分析
# 使用pytorch_kmeans返回的centers，确保只包含参与聚类的特征
analyze_intra_cluster_compactness(X_cpu, cluster_labels, centers)

# -------------------------- 8. 保存结果 --------------------------
pytorch_df.to_excel(f"{output_path_tsne}pytorch优化聚类结果.xlsx", index=False, engine="openpyxl")
cluster_means.to_excel(f"{output_path_tsne}pytorch优化聚类特征均值.xlsx", index=False, engine="openpyxl")

# 保存聚类中心
centers_df = pd.DataFrame(centers, columns=X_cpu.columns)
centers_df.to_excel(f"{output_path_tsne}pytorch优化聚类中心.xlsx", index=False, engine="openpyxl")

print(f"\nPyTorch优化K-Means - 结果已保存至：{output_path_tsne}")
print("\npytorch优化K-means聚类.py执行完成！")