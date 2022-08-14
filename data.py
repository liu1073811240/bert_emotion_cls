from datasets import list_datasets, load_dataset, load_from_disk

# print(list_datasets())  # 展示所有数据集名称

# dataset = load_dataset(path="seamew/ChnSentiCorp",split="train")
# print(dataset)

# 加载磁盘数据
dataset = load_from_disk("data/ChnSentiCorp")
print(dataset)

# 取出训练集
train_data = dataset["train"]
print(train_data)

# 查看数据
for data in train_data:
    print(data)
