from torch.utils.data import Dataset
from datasets import load_from_disk


class MyDataset(Dataset):
    def __init__(self, split):
        # self.dataset = load_dataset(path="seamew/ChnSentiCorp",split="train")
        # 从磁盘加载数据
        self.dataset = load_from_disk("data/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]

        return text, label


if __name__ == '__main__':
    dataset = MyDataset("validation")
    for data in dataset:
        print(data)

    print(len(dataset))
