from transformers import BertTokenizer

# 加载字典和分词工具
token = BertTokenizer.from_pretrained("bert-base-chinese")
# print(token)
'''
PreTrainedTokenizer(
name_or_path='bert-base-chinese', 
vocab_size=21128,   # 包含21128个词
model_max_len=512,  # 模型最大长度
is_fast=False, 
padding_side='right', 
truncation_side='right', 
special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})

# unk_token表示未识别出来的标记。
# special_tokens里面包括五个特殊字符。

'''

sents = [
    "酒店太旧了， 大堂感觉象三星级的， 房间也就是的好点的三星级的条件， 在青岛这样的酒店是绝对算不上四星标准， 早餐走了两圈也没有找到可以吃的， 太差了",
    "已经贴完了，又给小区的妈妈买了一套。最值得推荐",
    "屏幕大，本本薄。自带数字小键盘，比较少见。声音也还过得去。usb接口多，有四个。独显看高清很好。运行速度也还可以，性价比高！",
    "酒店环境很好 就是有一点点偏 交通不是很便利 去哪都需要达车 关键是不好打 酒店应该想办法解决一下"]

# 批量编码句子
out = token.batch_encode_plus(
    # batch_text_or_text_pairs=[sents[0], sents[1]],
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],  # 最多支持两对句子
    add_special_tokens=True,
    # 当句子长度大于max_length时,截断
    truncation=True,
    # 一律补零到max_length长度
    padding="max_length",
    max_length=30,
    # 可取值tf,pt,np,默认为返回list
    return_tensors=None,
    # 返回attention_mask
    return_attention_mask=True,
    # 返回token_type_ids
    return_token_type_ids=True,
    # 返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,
    # 返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    # return_offsets_mapping=True,
    # 返回length 标识长度
    return_length=True,
)
# input_ids 就是编码后的词
# token_type_ids 第一个句子和特殊符号的位置是0,第二个句子和特殊符号的位置是1。 但要在句子段的时候才会体现出来，这里不会显示出来。
# special_tokens_mask 特殊符号的位置是1,其他位置是0
# attention_mask pad的位置是0,其他位置是1
# length 返回句子长度
print(out)
for k, v in out.items():
    print(k, ":", v)
#
print(token.decode(out["input_ids"][0]), token.decode(out["input_ids"][1]))
print(len(out["input_ids"][0]))
print(len(out["attention_mask"][0]))

print('*' * 30)

# #获取字典
vocab = token.get_vocab()
print(type(vocab), len(vocab), "阳光" in vocab)  # <class 'dict'> 21128 False

# #添加新词
token.add_tokens(new_tokens=["阳光", "大地"])
# #添加新符号
token.add_special_tokens({"eos_token": "[EOS]"})

vocab = token.get_vocab()
print(type(vocab), len(vocab), "阳光" in vocab, "[EOS]" in vocab)  # <class 'dict'> 21131 True True
print(vocab["阳光"], vocab["大地"], vocab["[EOS]"])  # 21128 21129 21130

# #编码新句子
out = token.encode(
    text="阳光洒在大地上[EOS]",
    text_pair=None,
    truncation=True,
    padding="max_length",
    max_length=10,
    add_special_tokens=True,
    return_tensors=None
)
print(out)  # [101, 21128, 3818, 1762, 21129, 677, 21130, 102, 0, 0]
print(token.decode(out))  # [CLS] 阳光 洒 在 大地 上 [EOS] [SEP] [PAD] [PAD]
