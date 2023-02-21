# 训练P网络

import nets
import train
import torch

if __name__ == '__main__':
    net = nets.PNet()

    try :
        net.load_state_dict(torch.load('param/pnet.pt'))
        print("加载成功！")
    except:
        pass

    trainer = train.Trainer(net, 'param/pnet.pt', r"E:\celeba\12")      # 网络、保存参数、训练数据
    trainer.train()                                                     # 调用训练方法
