import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
from torch import nn
from torch.nn import functional as F
from model import AutoEncoder, VAE
from dataset import SplitedMNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

config = {
    'model': 'ae', # 'ae' or 'vae'
    'z_dim': 64,
    'batch': 16,
    'epochs': 100,
    'lr': 2.0e-5,
    'device': 'cuda:0'
}
device = config['device']
train_dataset = SplitedMNIST([2]) # 1を学習
train_loader = DataLoader(train_dataset, batch_size=config['batch'], shuffle=True)
test_dataset = SplitedMNIST([2,3], train=False) #1に8を混ぜてテスト
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
if config['model'] == 'ae':
    model = AutoEncoder(config['z_dim'])
else:
    model = VAE(config['z_dim'], device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config['lr'],
                             weight_decay=1e-5)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir="./logs")
model.cuda(device)

def loss_vae(x_h, x, mean, logvar):
    bce = F.binary_cross_entropy(x_h, x, reduction='sum')
    kl_d = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
    return bce, kl_d


def train():
    losses = []
    for epoch in range(config['epochs']):
        running_loss = 0.0
        for i, img in enumerate(train_loader, 1):
            x = img.view(img.size(0), -1).to(device)
            optimizer.zero_grad()
            if config['model'] == 'ae':
                x_h = model(x)
                loss = criterion(x, x_h)
            else:
                x_h, _, mean, logvar = model(x, device)
                bce, kl = loss_vae(x_h, x, mean, logvar)
                loss = bce + kl
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar("loss", loss.cpu().detach().numpy(), (epoch+1)*len(train_loader.dataset) + i)
        losses.append(running_loss/len(train_loader.dataset))
        print(f'epoch: {epoch}, loss: {losses[-1]:.4f}')
    writer.close()
    torch.save(model.state_dict(), 'models/model_ae_2.pth')
        

def test():
    if config['model'] == 'ae':
        model = AutoEncoder(config['z_dim'])
        model.load_state_dict(torch.load('models/model_ae_2.pth'))
    else:
        model = VAE(config['z_dim'], device)
        model.load_state_dict(torch.load('models/model_vae_1.pth'))
    model.cuda()
    for img in test_loader:
        x = img.view(img.size(0), -1).to(device)

        if config['model'] == 'ae':
            x_h = model(x)
        else:
            x_h, _, mean, logvar = model(x, device)
    return x, x_h

# def evaluate_vae(model, x, height=8, width=8, move=4):
#     x =torch.from_numpy(x).to('cuda:0')
#     x_h, _, mean, logvar = model(x)
#     bce = F.binary_cross_entropy(x_h, x, reduction='sum')
#     x = x.cpu().detach().numpy()
#     mean = mean.cpu().detach().numpy()
#     logvar = logvar.cpu().detach().numpy()
#     x = x.reshape(1,28,28)
#     score = 0
#     img = np.zeros((x.shape))
#     for i in range(int((x.shape[1]-height)/move)):
#         for j in range(int((x.shape[2]-width)/move)):
#             x_sub = x[0, i*move:i*move+height, j*move:j*move+width]
#             x_sub = x_sub.reshape(-1, height, width)

#             mean = mean.reshape(1,8,8)
#             logvar = logvar.reshape(1,8,8)
#             sigma = np.exp(logvar)

#             loss = 0
#             for k in range(height):
#                 for l in range(width):
#                     loss += 0.5 * (x_sub[0,k,l] - mean[0,k,l])**2 / sigma[0,k,l]
#             img[0, i*move:i*move+height, j*move:j*move+width] +=  loss
#             score = bce 
#     return img, score

def evaluate(x, x_h):
    x = x/2 + 0.5
    if config['model'] == 'ae':
        x_h = x_h/2 + 0.5
    x = x.cpu().detach().numpy()
    x_h = x_h.cpu().detach().numpy()
    n = 10
    plt.figure(figsize=(24, 12))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_h[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        diff_img = np.abs(x[i] - x_h[i])   
        score = sum(diff_img)
        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(diff_img.reshape(28, 28),cmap="Blues",norm=colors.LogNorm())
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        ax.set_xlabel(f'score = {score:.4f}')

    plt.savefig("save/result_ae_2_3.png")
    plt.show()
    plt.close()

        
if __name__ == "__main__":
    train()
    x, x_h = test()
    evaluate(x, x_h)
