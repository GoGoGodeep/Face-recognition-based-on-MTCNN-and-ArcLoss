# O网络训练
import torch
import nets
import train

if __name__ == '__main__':
    net = nets.ONet()
    try :
        net.load_state_dict(torch.load('param/onet.pt'))
        print("加载成功！")
    except:
        pass

    trainer = train.Trainer(net, 'param/onet.pt', r"E:/celeba/48")    # 网络，保存参数，训练数据；创建训器
    trainer.train()     # 调用训练器中的train方法