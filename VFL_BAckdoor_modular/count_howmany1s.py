def count_columns(file_path):
    col1_count = 0
    col2_count = 0
    both_count = 0
    target = "1.000000000000000000e+00"
    
    with open(file_path, 'r') as f:
        for line in f:
            # 去除空白並拆解
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            # 統計各列
            if tokens[0] == target:
                col1_count += 1
            if tokens[1] == target:
                col2_count += 1
            # 同時為 1 的行
            if tokens[0] == target and tokens[1] == target:
                both_count += 1
    return col1_count, col2_count, both_count

# 請修改成你的檔案路徑
file_path = "poisoned_voting_record.txt"
col1, col2, both = count_columns(file_path)

print(f"第一行總共有 {col1} 個 1")
print(f"第二行總共有 {col2} 個 1")
print(f"兩行同時為1的有 {both} 個")
