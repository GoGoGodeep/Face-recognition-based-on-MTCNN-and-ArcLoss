# 训练R网络

import nets
import train
import torch

if __name__ == '__main__':
    net = nets.RNet()

    try :
        net.load_state_dict(torch.load('param/rnet.pt'))
        print("加载成功！")
    except:
        pass

    trainer = train.Trainer(net, 'param/rnet.pt', r"E:\celeba\24") # 网络，保存参数，训练数据；创建训练器
    trainer.train()                                                    # 调用训练器中的方法
