import pandas as pd

df = pd.read_csv("./data/bank-market/bank-market.csv")

# 取得目前所有欄位名稱
cols = df.columns.tolist()

# 把 'duration' 先拿掉，再插入到第 8 個位置（索引 7）
cols.insert(7, cols.pop(cols.index('duration')))

# 依照新的順序重新排列 DataFrame
df = df[cols]

# 如果需要，存回檔案
df.to_csv("./data/bank-market/bank-market_reordered.csv", index=False)
