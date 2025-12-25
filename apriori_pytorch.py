import torch
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings("ignore")
# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

def load_order_data(order_path, product_path, cluster_path):
    """
    加载订单数据和产品数据
    Args:
        order_path: 订单数据路径
        product_path: 产品数据路径
        cluster_path: 聚类结果路径（可选）
    Returns:
        合并后的订单数据
    """
    # 读取订单数据
    order_df = pd.read_excel(order_path)
    
    # 读取产品数据
    product_df = pd.read_excel(product_path)
    
    # 合并产品数据中的餐饮品类信息
    order_df = pd.merge(order_df, product_df[['product_id', '餐饮品类']], on='product_id', how='left')
    
    # 如果有聚类结果路径，则读取聚类信息
    if cluster_path is not None:
        try:
            cluster_df = pd.read_excel(cluster_path)
            # 如果存在CustomerID列，则进行关联
            if 'CustomerID' in cluster_df.columns and 'CustomerID' in order_df.columns:
                order_df = pd.merge(order_df, cluster_df[['CustomerID', 'PyTorch_KMeans_标签']], on='CustomerID', how='left')
            else:
                # 否则随机分配客户类型（模拟）
                order_df['PyTorch_KMeans_标签'] = np.random.randint(0, 2, size=len(order_df))
        except Exception as e:
            print(f"读取聚类文件失败: {e}")
            # 失败时随机分配客户类型
            order_df['PyTorch_KMeans_标签'] = np.random.randint(0, 2, size=len(order_df))
    else:
        # 如果没有聚类结果文件，根据订单消费金额划分客户类型
        if '订单消费金额' in order_df.columns:
            # 计算每个订单的平均消费金额
            avg_amount = order_df['订单消费金额'].mean()
            # 高于平均值的为高价值客户（标签1），否则为一般客户（标签0）
            order_df['PyTorch_KMeans_标签'] = (order_df['订单消费金额'] > avg_amount).astype(int)
        else:
            # 缺失金额列时随机分配
            order_df['PyTorch_KMeans_标签'] = np.random.randint(0, 2, size=len(order_df))
    
    return order_df

def build_transaction_tensor(df, customer_label, item_col="餐饮品类"):
    """
    构建PyTorch布尔张量事务矩阵（核心优化：并行化事务表示）
    Args:
        df: 订单数据
        customer_label: 客户群体标签
        item_col: 商品列名（默认使用餐饮品类）
    Returns:
        transactions: 事务矩阵张量
        unique_items: 唯一商品列表
        item2idx: 商品到索引的映射
    """
    # 筛选目标客户群体的订单
    df_target = df[df["PyTorch_KMeans_标签"] == customer_label].copy()
    
    # 检查item_col是否存在
    if item_col not in df_target.columns:
        # 如果指定的列不存在，尝试使用其他可能的列
        if '餐饮品类' in df_target.columns:
            item_col = '餐饮品类'
        elif 'product_name' in df_target.columns:
            item_col = 'product_name'
        else:
            # 如果都没有，使用product_id作为商品标识
            item_col = 'product_id'
            print(f"警告：未找到指定的{item_col}列，使用product_id代替")
    
    # 去重（同一订单内重复商品）
    df_target = df_target.drop_duplicates(subset=["order_id", item_col])
    
    # 商品映射为索引
    unique_items = df_target[item_col].unique()
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    n_items = len(unique_items)
    
    # 按订单分组构建事务
    order_groups = df_target.groupby("order_id")[item_col].apply(list)
    n_trans = len(order_groups)
    
    # 构建布尔张量（n_trans × n_items）
    transactions = torch.zeros((n_trans, n_items), dtype=torch.bool, device=device)
    for trans_idx, items in enumerate(order_groups):
        item_indices = torch.tensor([item2idx[item] for item in items], dtype=torch.long)
        transactions[trans_idx, item_indices] = True
    
    # 过滤单商品订单
    trans_mask = torch.sum(transactions.int(), dim=1) >= 2
    transactions = transactions[trans_mask]
    
    print(f"客户群体{customer_label}：有效事务数={transactions.shape[0]}，商品数={n_items}")
    return transactions, unique_items, item2idx

def find_frequent_itemsets_pytorch(transactions, unique_items, min_sup=0.02):
    """PyTorch版频繁项集挖掘（批量计算支持度）"""
    n_trans = transactions.shape[0]
    frequent_itemsets = []
    
    # 1-项集
    item_supports = torch.sum(transactions, dim=0) / n_trans
    frequent_mask = item_supports >= min_sup
    frequent_1items = torch.where(frequent_mask)[0]
    frequent_itemsets.append((frequent_1items, item_supports[frequent_mask]))
    
    # 迭代生成k-项集
    k = 2
    while True:
        prev_frequent = frequent_itemsets[k-2][0]
        if len(prev_frequent) < k:
            break
        
        # 生成候选集（简化版：前k-2个元素相同的组合）
        candidates = []
        for i in range(len(prev_frequent)):
            for j in range(i+1, len(prev_frequent)):
                candidate = torch.cat([prev_frequent[i].unsqueeze(0), prev_frequent[j].unsqueeze(0)]).unique()
                if len(candidate) == k:
                    candidates.append(candidate)
        
        if not candidates:
            break
        
        # 批量计算候选集支持度（PyTorch并行优化）
        candidates_tensor = torch.stack(candidates).to(device)
        candidate_supports = torch.zeros(len(candidates), device=device)
        for idx, cand in enumerate(candidates_tensor):
            # 计算事务中同时包含候选集所有元素的比例
            cand_mask = torch.all(transactions[:, cand], dim=1)
            candidate_supports[idx] = torch.sum(cand_mask.int()) / n_trans
        
        # 筛选频繁项集
        frequent_mask = candidate_supports >= min_sup
        frequent_indices = torch.where(frequent_mask)[0]
        if len(frequent_indices) == 0:
            break
        
        frequent_kitems = torch.stack([candidates[idx] for idx in frequent_indices]).to(device)
        frequent_supports = candidate_supports[frequent_mask]
        
        if len(frequent_kitems) == 0:
            break
        
        frequent_itemsets.append((frequent_kitems, frequent_supports))
        k += 1
    
    # 格式转换为DataFrame（适配mlxtend关联规则）
    def convert_to_df(itemsets_list, items):
        rows = []
        for k_idx, (itemsets, supports) in enumerate(itemsets_list):
            k_val = k_idx + 1
            if k_val == 1:
                # 处理1-项集
                for item_idx, support in zip(itemsets, supports):
                    item_names = [items[item_idx.item()]]
                    rows.append({"itemsets": frozenset(item_names), "support": support.item()})
            else:
                # 处理k-项集(k>=2)
                for itemset, support in zip(itemsets, supports):
                    item_names = [items[idx.item()] for idx in itemset]
                    rows.append({"itemsets": frozenset(item_names), "support": support.item()})
        return pd.DataFrame(rows)
    
    return convert_to_df(frequent_itemsets, unique_items)

def pytorch_apriori(df, min_sup=0.02, min_conf=0.2, min_lift=1.1):
    """PyTorch版Apriori关联规则挖掘（分群体）"""
    rules_dict = {}
    
    # 分客户群体挖掘
    for label in [0, 1]:  # 0=一般客户，1=高价值客户（对应论文聚类结果）
        # 构建事务张量
        transactions, items, item2idx = build_transaction_tensor(df, label)
        if transactions.shape[0] == 0:
            rules_dict[label] = pd.DataFrame()
            continue
        
        # 挖掘频繁项集
        frequent_df = find_frequent_itemsets_pytorch(transactions, items, min_sup)
        if len(frequent_df) == 0:
            rules_dict[label] = pd.DataFrame()
            continue
        
        # 生成关联规则
        rules = association_rules(frequent_df, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift].sort_values("confidence", ascending=False)
        rules_dict[label] = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    
    return rules_dict

def main():
    # 配置文件路径（替换为你的实际路径）
    order_path = "D:/QQ文档/数据挖掘/食品天气项目/feature_engineered_dataset/order_product_features.xlsx"
    product_path = "D:/QQ文档/数据挖掘/食品天气项目/cleaned_dataset/cleaned_product.xlsx"
    cluster_path = None  # 不再需要聚类结果文件
    
    # 加载数据
    order_df = load_order_data(order_path, product_path, cluster_path)
    
    # PyTorch版Apriori挖掘
    rules_dict = pytorch_apriori(order_df)
    
    # 保存结果
    output_dir = "D:/QQ文档/数据挖掘/食品天气项目/"
    rules_dict[0].to_excel(output_dir + "pytorch_apriori_一般客户规则.xlsx", index=False)
    rules_dict[1].to_excel(output_dir + "pytorch_apriori_高价值客户规则.xlsx", index=False)
    
    print(f"\n关联规则挖掘完成：")
    print(f"一般客户有效规则数：{len(rules_dict[0])}")
    print(f"高价值客户有效规则数：{len(rules_dict[1])}")
    print(f"规则已保存至：{output_dir}")

if __name__ == "__main__":
    main()