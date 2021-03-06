import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_gen import DatasetFromFolder
import itertools
import os
from models import ResnetGenerator, Discriminator, device, PatchLoss
from utils import AverageMeter, visualize, weights_init_normal, clip_weight

root = 'data/selfie2anime'

if_train_d = True
light = True
d_train_freq = 10
clip = 0.01
print_freq = 100
epochs = 1000
lr = 0.0001
weight_decay = 0.0001
batch_size = 1
test_batch_size = 1
input_channel = 3
output_channel = 3
ngf = 64
ndf = 64
g_layer = 4
d_layer = 6
check = 'best_checkpoint.tar'
weight_gan = 100
weight_cycle = 10
weight_identity = 10
weight_cam = 10

train_set = DatasetFromFolder(root, 'train')
train_loader = DataLoader(train_set, batch_size, True)

test_set = DatasetFromFolder(root, 'test')
test_loader = DataLoader(test_set, 1, True)

if os.path.exists(check):
    print('load checkpoint')
    checkpoint = torch.load(check)
    netg_a2b = checkpoint[0]
    netg_b2a = checkpoint[1]
    netd_a = checkpoint[2]
    netd_b = checkpoint[3]
else:
    print('train from init')
    netg_a2b = ResnetGenerator(input_channel, output_channel, ngf=ngf, n_blocks=g_layer, light=light).to(device)
    netg_b2a = ResnetGenerator(input_channel, output_channel, ngf=ngf, n_blocks=g_layer, light=light).to(device)
    netd_a = Discriminator(input_channel, ndf=ndf, n_layers=d_layer).to(device)
    netd_b = Discriminator(input_channel, ndf=ndf, n_layers=d_layer).to(device)

criterionMSE = PatchLoss(nn.MSELoss()).to(device)
criterionL1 = nn.L1Loss().to(device)
criterionBCE = PatchLoss(nn.BCEWithLogitsLoss()).to(device)

optimzer_g = torch.optim.Adam(itertools.chain(netg_b2a.parameters(), netg_a2b.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
optimzer_d = torch.optim.Adam(itertools.chain(netd_a.parameters(), netd_b.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
if not os.path.exists(check):
    print('init param')
    weights_init_normal(optimzer_g)
    weights_init_normal(optimzer_d)


def train():
    for epoch in range(epochs):
        avg_loss_g_a = AverageMeter()
        avg_loss_g_b = AverageMeter()
        avg_loss_d_a = AverageMeter()
        avg_loss_d_b = AverageMeter()
        min_loss_g = float('inf')
        min_loss_d = float('inf')
        for i, data in enumerate(train_loader):
            img_a, img_b = data[0].to(device), data[1].to(device)

            #### update generator
            optimzer_g.zero_grad()

            fake_b2b, fake_b2b_cam_logit, _ = netg_a2b(img_b)
            fake_a2a, fake_a2a_cam_logit, _ = netg_b2a(img_a)

            fake_a2b, fake_a2b_cam_logit, _ = netg_a2b(img_a)
            fake_b2a, fake_b2a_cam_logit, _ = netg_b2a(img_b)

            recover_a, _, _ = netg_b2a(fake_a2b)
            recover_b, _, _ = netg_a2b(fake_b2a)

            pred_a, pred_a_cam_logit, _ = netd_a(fake_b2a)
            pred_b, pred_b_cam_logit, _ = netd_b(fake_a2b)

            # gan loss
            loss_gd_a = criterionMSE(pred_a, True)
            loss_gd_a_cam = criterionMSE(pred_a_cam_logit, True)
            loss_gd_b = criterionMSE(pred_b, True)
            loss_gd_b_cam = criterionMSE(pred_b_cam_logit, True)

            # identity loss
            loss_id_b = criterionL1(img_b, fake_b2b)
            loss_id_a = criterionL1(img_a, fake_a2a)

            # cycle loss
            loss_cycle_a = criterionL1(recover_a, img_a)
            loss_cycle_b = criterionL1(recover_b, img_b)

            # cam loss
            cam_loss_a = criterionBCE(fake_b2a_cam_logit, True) + criterionBCE(fake_a2a_cam_logit, False)
            cam_loss_b = criterionBCE(fake_a2b_cam_logit, True) + criterionBCE(fake_b2b_cam_logit, False)

            loss_G_a = (loss_gd_a + loss_gd_a_cam) * weight_gan + loss_cycle_a * weight_cycle + loss_id_a * weight_identity + cam_loss_a * weight_cam
            loss_G_b = (loss_gd_b + loss_gd_b_cam) * weight_gan + loss_cycle_b * weight_cycle + loss_id_b * weight_identity + cam_loss_b * weight_cam
            loss_G = loss_G_a + loss_G_b

            loss_G.backward()
            optimzer_g.step()

            avg_loss_g_a.update(loss_G_a)
            avg_loss_g_b.update(loss_G_b)

            if (i + 1) % d_train_freq == 0:
                #### update discriminator
                optimzer_d.zero_grad()

                pred_real_a, pred_real_a_cam_logit, _ = netd_a(img_a)
                pred_fake_a, pred_fake_a_cam_logit, _ = netd_a(fake_b2a.detach())

                pred_real_b, pred_real_b_cam_logit, _ = netd_b(img_b)
                pred_fake_b, pred_fake_b_cam_logit, _ = netd_b(fake_a2b.detach())

                # gan loss
                loss_d_a_real = criterionMSE(pred_real_a, True)
                loss_d_a_fake = criterionMSE(pred_fake_a, False)
                loss_d_a = loss_d_a_real + loss_d_a_fake

                loss_d_b_real = criterionMSE(pred_real_b, True)
                loss_d_b_fake = criterionMSE(pred_fake_b, False)
                loss_d_b = loss_d_b_real + loss_d_b_fake
                # cam loss
                loss_d_a_cam_real = criterionMSE(pred_real_a_cam_logit, True)
                loss_d_a_cam_fake = criterionMSE(pred_fake_a_cam_logit, False)
                loss_d_a_cam = loss_d_a_cam_real + loss_d_a_cam_fake

                loss_d_b_cam_real = criterionMSE(pred_real_b_cam_logit, True)
                loss_d_b_cam_fake = criterionMSE(pred_fake_b_cam_logit, False)
                loss_d_b_cam = loss_d_b_cam_real + loss_d_b_cam_fake

                loss_D_a = loss_d_a + loss_d_a_cam
                loss_D_b = loss_d_b + loss_d_b_cam
                avg_loss_d_a.update(loss_D_a)
                avg_loss_d_b.update(loss_D_b)
                loss_D = loss_D_a + loss_D_b
                if if_train_d:
                    loss_D.backward()
                    optimzer_d.step()

            if (i + 1) % print_freq == 0:
                print('##################   epoch {0} {1}/{2}  ####################'.format(epoch, i, train_loader.__len__()))
                print('generator loss a ', loss_gd_a.detach().cpu().numpy(), loss_gd_a_cam.detach().cpu().numpy(), loss_cycle_a.detach().cpu().numpy(),
                      loss_id_a.detach().cpu().numpy(), cam_loss_a.detach().cpu().numpy())
                print('generator loss b ', loss_gd_b.detach().cpu().numpy(), loss_gd_b_cam.detach().cpu().numpy(), loss_cycle_b.detach().cpu().numpy(),
                      loss_id_b.detach().cpu().numpy(), cam_loss_b.detach().cpu().numpy())
                print('discriminator loss a', loss_d_a_real.detach().cpu().numpy(), loss_d_a_fake.detach().cpu().numpy(), loss_d_a_cam_real.detach().cpu().numpy(),
                      loss_d_a_cam_fake.detach().cpu().numpy())
                print('discriminator loss b', loss_d_b_real.detach().cpu().numpy(), loss_d_b_fake.detach().cpu().numpy(), loss_d_b_cam_real.detach().cpu().numpy(),
                      loss_d_b_cam_fake.detach().cpu().numpy())
                print('loss: avg_loss_d_a {1:.3f} avg_loss_d_b {2:.3f} avg_loss_g_d_a {3:.3f} avg_loss_g_d_b {4:.3f}'
                      .format(0, avg_loss_d_a.avg, avg_loss_d_b.avg, avg_loss_g_a.avg, avg_loss_g_b.avg))
        if loss_G < min_loss_g and loss_D < min_loss_d:
            min_loss_g = loss_G
            min_loss_d = loss_D
            torch.save((netg_a2b, netg_b2a, netd_a, netd_b), check)

        visualize(netg_a2b, netg_b2a, test_loader)


if __name__ == '__main__':
    train()
