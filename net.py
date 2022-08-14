# 加载预训练模型
from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained("bert-base-chinese").to(DEVICE)

# print(pretrained)


# 定义下游任务模型（讲主干网络所提取的特征分类）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():  # 主干网络的权重不参与训练更新。只训练上面一层网络
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
