import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from transformers import AdamW
from transformers.optimization import get_scheduler
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

#定义在gpu上训练
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#加载gpt2模型与tokenizer
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)

#加载古诗词
# 设定自己的训练集格式
class MyDataset(Dataset):

    def __init__(self):
        data = pd.read_csv('data/唐.csv')
        data = data['内容']
        data = data.str.strip()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data.iloc[i]
#实例化
dataset = MyDataset()

def collate_fn(data):
    '''
    用于帮文本token化
    '''
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,
                                       truncation=True,
                                       max_length=512,
                                       return_tensors='pt')
    data['labels'] = data['input_ids'].clone()

    return data

batch_size = 3
#加载好数据
loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)

# 接下来进行训练
# 定义优化器为AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)
# 定义学习率调度器，该调度器用于改变学习率大小,学习率逐层递减
scheduler = get_scheduler(name='linear',
                          num_warmup_steps=0,
                          num_training_steps=len(loader),
                          optimizer=optimizer)
model.train()
epochs = 10
loss_road = []
lr_road = []
#每个epoch
for epoch in range(epochs):
    sum_loss = 0.0
    for data in tqdm.tqdm(loader):
        #将数据放入gpu中，因为是字典形式只能一个一个放
        for k in data.keys():
            data[k] = data[k].to(device)
        #得到结果y_pred
        y_pred = model(**data)
        loss = y_pred['loss']
        #反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #梯度优化
        optimizer.step()
        #梯度清零
        optimizer.zero_grad()
#        model.zero_grad()
        #计算一些loss
        sum_loss += loss.item()
    # 获取当前的lr大小
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    # 每个epoch更新调度器
    scheduler.step()
    train_loss  = sum_loss / ((dataset.__len__() / batch_size))
    #进行输出
    print(f"epoch: {epoch}, train loss: {train_loss:.4f}, learning rate: {lr:.4f}")
    # 记录loss变化
    loss_road.append(train_loss)
    # 记录lr变化
    lr_road.append(lr)
    # 保存模型
    name = 'ckpt_epoch_'+ str(epoch) + '.model'
    torch.save(model, name)

# 绘制loss折线图
plt.figure(1)
plt.plot(loss_road, color='blue', linestyle='-', linewidth=2)
# 设置图表标题和轴标签
plt.title('Loss road')
plt.xlabel('epoch')
plt.ylabel('Loss')
# 保存图片到本地
plt.savefig('loss.png')
plt.show()

# 绘制bleu折线图
plt.figure(2)
plt.plot(lr_road, color='green', linestyle='-', linewidth=2)
# 设置图表标题和轴标签
plt.title('lr road')
plt.xlabel('epoch')
plt.ylabel('lr')
# 保存图片到本地
plt.savefig('lr.png')
plt.show()