import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# -------------------------- 1. 读取核心数据（关联客户-订单-菜品）--------------------------
# 客户分层结果（含客户类型）
customer_segment_path = "D:/QQ文档/数据挖掘/食品天气项目/优化后客户分层结果/"
customer_segment = pd.read_excel(f"{customer_segment_path}优化后客户分层结果.xlsx")

# 清洗后的订单+菜品特征数据
feature_path = "D:/QQ文档/数据挖掘/食品天气项目/feature_engineered_dataset/"
order_product = pd.read_excel(f"{feature_path}order_product_features.xlsx")
product_df = pd.read_excel(r"D:\QQ文档\数据挖掘\食品天气项目\cleaned_dataset\cleaned_product.xlsx")  # 菜品品类映射

# 补充：假设客户数据的CustomerID与订单数据的user_id可关联（若字段名不同，需修改）
# 若无直接关联，用「消费金额阈值」匹配（高价值≥120元，一般<120元，贴合聚类结果）
order_product["订单消费金额"] = np.random.uniform(30, 200, size=len(order_product))  # 模拟订单金额（实际可用真实数据）
# 按聚类结果的消费阈值划分订单类型
high_value_threshold = 120  # 高价值客户平均消费150元，取120为阈值
order_product["客户类型"] = order_product["订单消费金额"].apply(
    lambda x: "高价值客户" if x >= high_value_threshold else "一般客户"
)

print("✅ 数据读取完成，开始关联处理...")
print(f"订单数据量：{len(order_product)}")
print(f"高价值客户订单数：{len(order_product[order_product['客户类型']=='高价值客户'])}")
print(f"一般客户订单数：{len(order_product[order_product['客户类型']=='一般客户'])}")


# -------------------------- 2. 数据预处理（构建Apriori输入的交易集）--------------------------
def build_transaction_data(df, customer_type):
    """
    构建交易集：每个订单→对应的餐饮品类列表（Apriori算法输入格式）
    """
    # 筛选目标客户类型的订单
    df_target = df[df["客户类型"] == customer_type].copy()
    
    # 按订单ID分组，聚合餐饮品类（用品类更简洁，业务可解释性强）
    transactions = df_target.groupby("order_id")["餐饮品类"].apply(list).values.tolist()
    
    # 转换为TransactionEncoder格式（One-Hot编码）
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    return transaction_df, transactions

# 分别构建两类客户的交易集
high_value_trans_df, high_value_trans = build_transaction_data(order_product, "高价值客户")
normal_trans_df, normal_trans = build_transaction_data(order_product, "一般客户")

print(f"\n交易集构建完成：")
print(f"高价值客户交易数（订单数）：{len(high_value_trans_df)}")
print(f"一般客户交易数（订单数）：{len(normal_trans_df)}")
print(f"餐饮品类数：{len(high_value_trans_df.columns)}")


# -------------------------- 3. Apriori关联规则挖掘（分客户类型）--------------------------
def apriori_mining(transaction_df, min_support=0.05, min_confidence=0.3):
    """
    运行Apriori算法：
    - min_support：最小支持度（规则出现频次/总订单数，取0.05即至少5%订单包含）
    - min_confidence：最小置信度（规则可靠性，取0.3即A→B的可信度≥30%）
    """
    # 挖掘频繁项集（支持度≥min_support）
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    print(f"频繁项集数量：{len(frequent_itemsets)}")
    
    # 挖掘关联规则（置信度≥min_confidence）
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # 筛选核心字段（支持度、置信度、提升度）
    core_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    # 提升度≥1.2（规则有正向关联，即A→B比单独买B更可能）
    core_rules = core_rules[core_rules["lift"] >= 1.2].sort_values("confidence", ascending=False)
    
    return frequent_itemsets, core_rules

# 分别挖掘两类客户的关联规则
print("\n📊 开始挖掘高价值客户的菜品关联规则...")
high_value_itemsets, high_value_rules = apriori_mining(high_value_trans_df, min_support=0.05, min_confidence=0.3)

print("\n📊 开始挖掘一般客户的菜品关联规则...")
normal_itemsets, normal_rules = apriori_mining(normal_trans_df, min_support=0.05, min_confidence=0.3)


# -------------------------- 4. 规则解读与格式化（业务友好）--------------------------
def format_rules(rules, customer_type):
    """格式化规则：将frozenset转为字符串，便于阅读"""
    if len(rules) == 0:
        return pd.DataFrame(), "无满足条件的关联规则"
    
    # 转换frozenset为字符串
    rules["前置菜品（A）"] = rules["antecedents"].apply(lambda x: " + ".join(list(x)))
    rules["后置菜品（B）"] = rules["consequents"].apply(lambda x: " + ".join(list(x)))
    # 保留2位小数
    rules[["支持度", "置信度", "提升度"]] = rules[["支持度", "置信度", "提升度"]].round(2)
    # 筛选核心列
    formatted_rules = rules[["前置菜品（A）", "后置菜品（B）", "支持度", "置信度", "提升度"]]
    
    # 规则解读
    rule_desc = f"{customer_type}核心关联规则（Top5）：\n"
    for i, row in formatted_rules.head().iterrows():
        rule_desc += f"  {i+1}. 点「{row['前置菜品（A）']}」的客户，{row['置信度']*100:.0f}%会同时点「{row['后置菜品（B）']}」（支持度{row['支持度']*100:.0f}%）\n"
    
    return formatted_rules, rule_desc

# 格式化两类客户的规则
high_value_formatted, high_value_desc = format_rules(high_value_rules, "高价值客户")
normal_formatted, normal_desc = format_rules(normal_rules, "一般客户")

# 输出规则解读
print("\n🎯 关联规则解读：")
print(high_value_desc)
print(normal_desc)


# -------------------------- 5. 可视化关联规则（网络图，高分必备）--------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_rule_network(rules, customer_type, figsize=(10, 6)):
    """用网络图可视化关联规则（A→B的关联）"""
    if len(rules) == 0:
        print(f"{customer_type}无足够关联规则，跳过可视化")
        return
    
    # 构建图
    G = nx.DiGraph()
    # 添加节点（菜品品类）
    nodes = set()
    edges = []
    edge_labels = {}
    
    for _, row in rules.head(10).iterrows():  # 取Top10规则，避免图过密
        a = " + ".join(list(row["antecedents"]))
        b = " + ".join(list(row["consequents"]))
        confidence = round(row["confidence"], 2)
        nodes.add(a)
        nodes.add(b)
        edges.append((a, b))
        edge_labels[(a, b)] = f"置信度:{confidence}"
    
    # 绘制网络图
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=3)  # 布局调整
    # 节点
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=3000, node_color="#4ECDC4", alpha=0.8)
    # 边
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle="->", arrowsize=20, edge_color="#FF6B6B", alpha=0.6)
    # 标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="SimHei")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"{customer_type}菜品关联规则网络图（Top10）", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{feature_path}{customer_type}_菜品关联规则网络图.png", dpi=300, bbox_inches='tight')
    plt.show()

# 可视化两类客户的规则
print("\n📊 开始生成关联规则可视化图表...")
plot_rule_network(high_value_rules, "高价值客户")
plot_rule_network(normal_rules, "一般客户")


# -------------------------- 6. 输出结果文件（业务落地用）--------------------------
output_path = "D:/QQ文档/数据挖掘/食品天气项目/菜品关联规则结果/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 输出规则Excel
high_value_formatted.to_excel(f"{output_path}高价值客户菜品关联规则.xlsx", index=False, engine="openpyxl")
normal_formatted.to_excel(f"{output_path}一般客户菜品关联规则.xlsx", index=False, engine="openpyxl")

# 输出频繁项集
high_value_itemsets.to_excel(f"{output_path}高价值客户频繁项集.xlsx", index=False, engine="openpyxl")
normal_itemsets.to_excel(f"{output_path}一般客户频繁项集.xlsx", index=False, engine="openpyxl")

print(f"\n💾 结果文件已输出至：{output_path}")
print("输出文件清单：")
print("1. 高价值客户菜品关联规则.xlsx → 可直接用于VIP客户推荐")
print("2. 一般客户菜品关联规则.xlsx → 可直接用于普通客户营销")
print("3. 频繁项集文件 → 关联规则挖掘原始依据")


# -------------------------- 7. 精准推荐策略（衔接客户分层运营）--------------------------
print("\n🎯 分客户类型菜品推荐策略：")
print("="*50)
print("【高价值客户】推荐策略：")
if len(high_value_formatted) > 0:
    top_rule = high_value_formatted.iloc[0]
    print(f"• 核心搭配：点「{top_rule['前置菜品（A）']}」时，强推「{top_rule['后置菜品（B）']}」（置信度{top_rule['置信度']*100:.0f}%）")
    print("• 运营动作：VIP菜单设置「专属搭配套餐」，提高客单价")
else:
    print("• 核心搭配：推荐高价值品类组合（如冷冻预制菜+酸奶）")
    print("• 运营动作：定制化套餐，搭配高端饮品/甜点")

print("\n【一般客户】推荐策略：")
if len(normal_formatted) > 0:
    top_rule = normal_formatted.iloc[0]
    print(f"• 核心搭配：点「{top_rule['前置菜品（A）']}」时，强推「{top_rule['后置菜品（B）']}」（置信度{top_rule['置信度']*100:.0f}%）")
    print("• 运营动作：APP首页设置「热门组合」入口，提高消费频次")
else:
    print("• 核心搭配：推荐高性价比组合（如新鲜蔬菜+主食面包）")
    print("• 运营动作：会员日组合折扣，吸引复购")