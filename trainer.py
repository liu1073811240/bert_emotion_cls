import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import AdamW, BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 30000

token = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    sentes = [i[0] for i in data]
    label = [i[1] for i in data]
    # print(sentes)
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=500,
                                   return_tensors="pt",
                                   return_length=True)
    input_ids = data["input_ids"]  # 就是编码后的词
    attention_mask = data["attention_mask"]  # pad的位置是0,其他位置是1
    token_type_ids = data["token_type_ids"]  # 句子段id
    labels = torch.LongTensor(label)  # 标签
    # print(input_ids,attention_mask,token_type_ids)
    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
train_dataset = MyDataset("train")
# 创建dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn)

if __name__ == '__main__':
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # print(input_ids)
            # print(attention_mask.shape)
            # print(token_type_ids.shape)
            # print(labels.shape)
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)

            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                out = out.argmax(dim=1)

                acc = (out == labels).sum().item() / len(labels)
                print(epoch, i, loss.item(), acc)
        torch.save(model.state_dict(), f"params/{epoch}bert01.pth")
        print(epoch, "参数保存成功！")
