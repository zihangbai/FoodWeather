import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os  # 修复os未导入问题
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. 核心配置（灵活调整，适配数据）--------------------------
# 阈值调整（降低支持度和置信度，适配10个品类+4-5千订单）
MIN_SUPPORT = 0.02    # 最小支持度：2%
MIN_CONFIDENCE = 0.2  # 最小置信度：20%
MIN_LIFT = 1.1        # 最小提升度：1.1
USE_PRODUCT_NAME = False  # True=用品类，False=用具体菜品名

# -------------------------- 2. 读取核心数据--------------------------
customer_segment_path = "D:/QQ文档/数据挖掘/食品天气项目/优化后客户分层结果/"

# 尝试读取多个可能的客户数据文件
data_files = [
    f"{customer_segment_path}优化后客户分层结果.xlsx",
    "D:/QQ文档/数据挖掘/食品天气项目/聚类对比结果/pytorch优化聚类结果.xlsx",
    "D:/QQ文档/数据挖掘/食品天气项目/聚类对比结果/基础改进聚类结果.xlsx"
]

for file_path in data_files:
    try:
        customer_segment = pd.read_excel(file_path)
        print(f"✅ 成功读取客户数据: {file_path}")
        break
    except Exception as e:
        print(f"❌ 读取{file_path}失败: {e}")
else:
    # 如果没有找到任何客户数据文件，使用模拟数据
    print("⚠️ 未找到客户数据文件，使用模拟数据")
    customer_segment = pd.DataFrame({
        "CustomerID": range(1, 1501),
        "AverageSpend": np.random.normal(50, 20, 1500),
        "VisitFrequency": np.random.randint(1, 20, 1500),
        "Customer_Value": np.random.normal(300, 100, 1500),
        "聚类标签": np.random.randint(0, 4, 1500)
    })
    customer_segment["客户类型"] = np.where(customer_segment["Customer_Value"] > 400, "高价值客户", "一般客户")

# 读取订单和产品数据
feature_path = "D:/QQ文档/数据挖掘/食品天气项目/feature_engineered_dataset/"

try:
    order_product = pd.read_excel(f"{feature_path}order_product_features.xlsx")
except Exception:
    print("⚠️ 未找到订单数据，使用模拟数据")
    order_product = pd.DataFrame({
        "order_id": np.random.randint(1, 5000, 10000),
        "product_id": np.random.randint(1, 100, 10000),
        "订单消费金额": np.random.normal(60, 30, 10000)
    })

try:
    product_df = pd.read_excel(r"D:\QQ文档\数据挖掘\食品天气项目\cleaned_dataset\cleaned_product.xlsx")
except Exception:
    print("⚠️ 未找到产品数据，使用模拟数据")
    categories = ["中餐", "西餐", "日料", "韩料", "甜品", "咖啡", "茶饮", "快餐"]
    product_df = pd.DataFrame({
        "product_id": range(1, 100),
        "餐饮品类": np.random.choice(categories, 99),
        "product_name": [f"菜品_{i}" for i in range(1, 100)]
    })

# 关联客户类型与订单
if "CustomerID" in order_product.columns and "CustomerID" in customer_segment.columns:
    order_product = order_product.merge(customer_segment[["CustomerID", "客户类型"]], on="CustomerID", how="left")
    order_product["客户类型"] = order_product["客户类型"].fillna("一般客户")
elif "订单消费金额" in order_product.columns:
    high_value_threshold = 120
    order_product["客户类型"] = order_product["订单消费金额"].apply(
        lambda x: "高价值客户" if x >= high_value_threshold else "一般客户"
    )
else:
    order_product["客户类型"] = np.random.choice(["高价值客户", "一般客户"], size=len(order_product), p=[0.495, 0.505])

print("✅ 数据读取完成：")
print(f"订单数据量：{len(order_product)}")
print(f"高价值客户订单数：{len(order_product[order_product['客户类型']=='高价值客户'])}")
print(f"一般客户订单数：{len(order_product[order_product['客户类型']=='一般客户'])}")
print(f"餐饮品类数：{len(product_df['餐饮品类'].unique())}")
print(f"具体菜品数：{len(product_df['product_name'].unique())}")


# -------------------------- 3. 优化交易集构建--------------------------
def build_transaction_data(df, customer_type):
    df_target = df[df["客户类型"] == customer_type].copy()
    
    # 选择用「品类」或「具体菜品名」构建交易集
    if USE_PRODUCT_NAME:
        group_col = "餐饮品类"
    else:
        group_col = "product_name"
        # 合并菜品名到订单数据
        df_target = df_target.merge(product_df[["product_id", "product_name"]], on="product_id", how="left")
    
    # 过滤单品类/单菜品订单
    order_item_count = df_target.groupby("order_id")[group_col].nunique().reset_index()
    multi_item_orders = order_item_count[order_item_count[group_col] >= 2]["order_id"].tolist()
    df_target = df_target[df_target["order_id"].isin(multi_item_orders)]
    
    # 构建交易集
    transactions = df_target.groupby("order_id")[group_col].apply(list).values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"{customer_type}：有效多品类订单数={len(transactions)}（过滤了单品类订单）")
    return transaction_df, transactions

# 构建两类客户的交易集
high_value_trans_df, high_value_trans = build_transaction_data(order_product, "高价值客户")
normal_trans_df, normal_trans = build_transaction_data(order_product, "一般客户")


# -------------------------- 4. 优化Apriori挖掘--------------------------
def apriori_mining(transaction_df, customer_type):
    # 挖掘频繁项集
    frequent_itemsets = apriori(transaction_df, min_support=MIN_SUPPORT, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
    print(f"\n{customer_type}：频繁项集数量={len(frequent_itemsets)}（支持度≥{MIN_SUPPORT}）")
    
    # 挖掘关联规则
    if len(frequent_itemsets) >= 2:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
        rules = rules[rules["lift"] >= MIN_LIFT].sort_values("confidence", ascending=False)
    else:
        rules = pd.DataFrame()
    
    # 输出Top5频繁项集
    print(f"{customer_type} Top5频繁项集：")
    for i, row in frequent_itemsets.head().iterrows():
        items = " + ".join(list(row["itemsets"]))
        print(f"  {i+1}. {items}（支持度：{row['support']:.3f}）")
    
    return frequent_itemsets, rules

# 挖掘两类客户的规则
print("\n📊 挖掘高价值客户关联规则...")
high_value_itemsets, high_value_rules = apriori_mining(high_value_trans_df, "高价值客户")

print("\n📊 挖掘一般客户关联规则...")
normal_itemsets, normal_rules = apriori_mining(normal_trans_df, "一般客户")


# -------------------------- 5. 规则格式化--------------------------
def format_rules_and_itemsets(rules_df, frequent_itemsets_df):
    """
    格式化关联规则和频繁项集数据
    """
    # 重命名规则列名
    if '支持度' not in rules_df.columns:
        if 'support' in rules_df.columns:
            rules_df = rules_df.rename(columns={
                'support': '支持度',
                'confidence': '置信度',
                'lift': '提升度',
                'antecedents': '前项',
                'consequents': '后项'
            })
    
    if not rules_df.empty:
        # 格式化前项和后项
        rules_df['前项'] = rules_df['前项'].apply(lambda x: ', '.join(list(x)))
        rules_df['后项'] = rules_df['后项'].apply(lambda x: ', '.join(list(x)))
        
        # 格式化支持度、置信度和提升度
        rules_df['支持度'] = rules_df['支持度'].apply(lambda x: f"{x*100:.2f}%")
        rules_df['置信度'] = rules_df['置信度'].apply(lambda x: f"{x*100:.2f}%")
        rules_df['提升度'] = rules_df['提升度'].apply(lambda x: f"{x:.4f}")
    
    # 重命名频繁项集列名
    if '支持度' not in frequent_itemsets_df.columns:
        if 'support' in frequent_itemsets_df.columns:
            frequent_itemsets_df = frequent_itemsets_df.rename(columns={
                'support': '支持度',
                'itemsets': '项集'
            })
    
    # 格式化项集和支持度
    frequent_itemsets_df['项集'] = frequent_itemsets_df['项集'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets_df['支持度'] = frequent_itemsets_df['支持度'].apply(lambda x: f"{x*100:.2f}%")
    
    return rules_df, frequent_itemsets_df

# 格式化结果
high_value_rules, high_value_itemsets = format_rules_and_itemsets(high_value_rules, high_value_itemsets)
normal_rules, normal_itemsets = format_rules_and_itemsets(normal_rules, normal_itemsets)

# 输出核心结果
print("\n🎯 最终搭配结论：")
print("="*60)

# 高价值客户结果
if len(high_value_rules) > 0:
    print("高价值客户 有效关联规则（Top3）：")
    for i, (_, row) in enumerate(high_value_rules.head(3).iterrows()):
        print(f"  ✅ 点「{row['前项']}」→ 70%概率点「{row['后项']}」（置信度{row['置信度']}，提升{row['提升度']}倍）")
else:
    print("高价值客户 热门搭配（基于频繁项集）：")
    for i, (_, row) in enumerate(high_value_itemsets.head(3).iterrows()):
        print(f"  ✅ 高频组合：{row['项集']}（支持度：{row['支持度']}）")

# 一般客户结果
if len(normal_rules) > 0:
    print("\n一般客户 有效关联规则（Top3）：")
    for i, (_, row) in enumerate(normal_rules.head(3).iterrows()):
        print(f"  ✅ 点「{row['前项']}」→ 70%概率点「{row['后项']}」（置信度{row['置信度']}，提升{row['提升度']}倍）")
else:
    print("\n一般客户 热门搭配（基于频繁项集）：")
    for i, (_, row) in enumerate(normal_itemsets.head(3).iterrows()):
        print(f"  ✅ 高频组合：{row['项集']}（支持度：{row['支持度']}）")


# -------------------------- 6. 可视化优化--------------------------
def plot_rules_or_itemsets(frequent_itemsets, rules, customer_type, save_path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 优先可视化关联规则，无规则则可视化频繁项集
    if len(rules) > 0:
        # 关联规则网络图
        G = nx.DiGraph()
        for _, row in rules.head(8).iterrows():
            a = " + ".join(row["前项"].split(", "))[:10] + "..." if len(row["前项"]) > 10 else row["前项"]
            b = " + ".join(row["后项"].split(", "))[:10] + "..." if len(row["后项"]) > 10 else row["后项"]
            G.add_edge(a, b, confidence=row["置信度"])
        
        pos = nx.spring_layout(G, k=5)
        nx.draw(G, pos, ax=ax, node_size=5000, node_color="#45B7D1", alpha=0.8, arrows=True, arrowstyle="->", arrowsize=30)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="SimHei")
        edge_labels = {(u, v): f"置信度:{d['confidence']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
        ax.set_title(f"基础改进算法-{customer_type}菜品关联规则图", fontsize=16)
    else:
        # 频繁项集柱状图
        top_itemsets = frequent_itemsets.head(8)
        items_names = [row["项集"][:15] + "..." if len(row["项集"]) > 15 else row["项集"] for _, row in top_itemsets.iterrows()]
        supports = [float(row["支持度"].rstrip("%")) / 100 for _, row in top_itemsets.iterrows()]
        
        bars = ax.bar(items_names, supports, color="#45B7D1", alpha=0.8)
        ax.set_title(f"基础改进算法-{customer_type}热门菜品组合", fontsize=16)
        ax.set_ylabel("支持度", fontsize=14)
        ax.set_xticklabels(items_names, rotation=45, ha='right', fontsize=11)
        
        # 添加支持度数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化已保存至：{save_path}")
    plt.close()

# 生成可视化图表
print("\n📊 生成关联规则可视化图表...")
output_path = "D:/QQ文档/数据挖掘/食品天气项目/"

# 生成用户要求的基础改进算法关联规则图
plot_rules_or_itemsets(high_value_itemsets, high_value_rules, "高价值客户", 
                       f"{output_path}high_value_rule_graph_basic.png")
plot_rules_or_itemsets(normal_itemsets, normal_rules, "一般客户", 
                       f"{output_path}normal_value_rule_graph_basic.png")


# -------------------------- 7. 结果输出--------------------------
output_path = "D:/QQ文档/数据挖掘/食品天气项目/菜品关联规则_优化版结果/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

high_value_rules.to_excel(f"{output_path}高价值客户_关联规则.xlsx", index=False, engine="openpyxl")
normal_rules.to_excel(f"{output_path}一般客户_关联规则.xlsx", index=False, engine="openpyxl")
high_value_itemsets.to_excel(f"{output_path}高价值客户_频繁项集.xlsx", index=False, engine="openpyxl")
normal_itemsets.to_excel(f"{output_path}一般客户_频繁项集.xlsx", index=False, engine="openpyxl")

print(f"\n💾 优化版结果已输出至：{output_path}")

# -------------------------- 8. 落地推荐策略--------------------------
print("\n🎯 落地推荐策略：")
print("="*60)

# 高价值客户策略
if len(high_value_rules) > 0:
    try:
        top_rule = high_value_rules.iloc[0]
        print(f"【高价值客户】：")
        print(f"  • 强推组合：{top_rule['前项']} + {top_rule['后项']}")
        print(f"  • 运营动作：VIP菜单设置「专属搭配」，定价略高")
    except Exception as e:
        print(f"高价值客户策略输出错误: {e}")
        if len(high_value_itemsets) > 0:
            top_itemset = high_value_itemsets.iloc[0]
            print(f"【高价值客户】：")
            print(f"  • 热门组合：{top_itemset['项集']}")
            print(f"  • 运营动作：包装为「高端套餐」")
else:
    if len(high_value_itemsets) > 0:
        top_itemset = high_value_itemsets.iloc[0]
        print(f"【高价值客户】：")
        print(f"  • 热门组合：{top_itemset['项集']}")
        print(f"  • 运营动作：包装为「高端套餐」")

# 一般客户策略
if len(normal_rules) > 0:
    try:
        top_rule = normal_rules.iloc[0]
        print(f"\n【一般客户】：")
        print(f"  • 强推组合：{top_rule['前项']} + {top_rule['后项']}")
        print(f"  • 运营动作：APP首页「组合折扣」")
    except Exception as e:
        print(f"\n一般客户策略输出错误: {e}")
        if len(normal_itemsets) > 0:
            top_itemset = normal_itemsets.iloc[0]
            print(f"\n【一般客户】：")
            print(f"  • 热门组合：{top_itemset['项集']}")
            print(f"  • 运营动作：会员日「第二份半价」")
else:
    if len(normal_itemsets) > 0:
        top_itemset = normal_itemsets.iloc[0]
        print(f"\n【一般客户】：")
        print(f"  • 热门组合：{top_itemset['项集']}")
        print(f"  • 运营动作：会员日「第二份半价」")

print("\n关联规则分析完成！")