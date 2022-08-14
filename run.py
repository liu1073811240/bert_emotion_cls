import torch
from net import Model
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
names = ["负向评价", "正向评价"]
print(DEVICE)
model = Model().to(DEVICE)

token = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    sentes = []
    sentes.append(data)
    # print(sentes)
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sentes,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=500,
                                   return_tensors="pt",
                                   return_length=True)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]

    # print(input_ids,attention_mask,token_type_ids)
    return input_ids, attention_mask, token_type_ids


def test():
    model.load_state_dict(torch.load("params/1bert01.pth"))
    model.eval()
    while True:
        data = input("请输入测试数据(输入'q'退出)：")
        if data == "q":
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：", names[out], "\n")


if __name__ == '__main__':
    test()
