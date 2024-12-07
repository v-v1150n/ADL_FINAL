import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 讀取 CSV 檔案
file_path = "result.csv"  # 替換為您的 CSV 檔案路徑
df = pd.read_csv(file_path)

# 確保四個指標的列名稱
discrete_metrics = ["context_recall"]  # 離散值
continuous_metrics = ["factual_correctness", "semantic_similarity", "faithfulness"]  # 連續值

# 去除列名稱中的多餘空格
df.columns = df.columns.str.strip()

# 繪製 Context Recall 和 Faithfulness 的值分布條形圖
for col in discrete_metrics:
    if col in df.columns:
        # 計算每個值的總數
        counts = df[col].value_counts().sort_index()  # 確保 0 和 1 的順序
        counts = counts.reindex([0, 1], fill_value=0)  # 確保列出所有值

        # 繪製條形圖
        plt.figure(figsize=(6, 4))
        plt.bar(counts.index, counts.values, color=['red', 'green'], alpha=0.8)
        plt.xticks([0, 1], labels=['0', '1'])  # 設定 X 軸標籤
        plt.title(f"Distribution of {col.replace('_', ' ').title()}")
        plt.xlabel("Value")
        plt.ylabel("Count")

        # 設定 Y 軸刻度為整數
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # 儲存圖表
        output_path = f"./figure/{col}_distribution_chart.png"
        plt.savefig(output_path, format="png")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        print(f"Column '{col}' is missing in the data.")

# 繪製 Factual Correctness 和 Semantic Similarity 的分段統計條形圖
bins = np.linspace(0, 1, 6)  # 分數範圍 [0, 1] 分成 5 段 (0-0.2, 0.2-0.4, ...)
bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]  # 標籤

for col in continuous_metrics:
    if col in df.columns:
        # 分段統計
        values = df[col]
        bin_counts = pd.cut(values, bins=bins, labels=bin_labels).value_counts(sort=False)

        # 繪製條形圖
        plt.figure(figsize=(8, 5))
        plt.bar(bin_labels, bin_counts.values, color='blue', alpha=0.8)
        plt.title(f"Distribution of {col.replace('_', ' ').title()}")
        plt.xlabel("Score Range")
        plt.ylabel("Count")

        # 設定 Y 軸刻度為整數
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # 儲存圖表
        output_path = f"./figure/{col}_binned_distribution_chart.png"
        plt.savefig(output_path, format="png")
        plt.close()
        print(f"Saved: {output_path}")
    else:
        print(f"Column '{col}' is missing in the data.")
