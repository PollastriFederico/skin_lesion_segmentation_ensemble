import os
import argparse
from torchvision import models
import torch
from torch import nn
from torch.autograd import Variable
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from random import choice
import csv
import cv2
from torchvision.utils import save_image

from my_code.grad_cam import GradCam

import my_code.isic.isic_ham_dataset as isic
import my_code.isic.isic_dataset as task1_isic

files_path = '/homes/my_d/data/istologia/files/'
data_path = '/homes/my_d/data/istologia/images'
parser = argparse.ArgumentParser()

parser.add_argument('--label', default='mesangiale', help='label to learn')
parser.add_argument('--classes', type=int, default=7, help='number of epochs to train')
parser.add_argument('--loadEpoch', type=int, default=0, help='load pretrained models')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=16, help='batch size during the training')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--thresh', type=float, default=0.5, help='number of data loading workers')
parser.add_argument('--epochs', type=int, default=41, help='number of epochs to train')
parser.add_argument('--size', type=int, default=512, help='size of images')
parser.add_argument('--n_splits', type=int, default=1, help='number of different dataset splits to test')
parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saving models')
parser.add_argument('--SRV', action='store_true', help='is training on remote server')
parser.add_argument('--from_scratch', action='store_true', help='not finetuning')

opt = parser.parse_args()
print(opt)


class MyResnet(nn.Module):
    def __init__(self, pretrained=False, num_classes=7):
        super(MyResnet, self).__init__()

        resnet = models.resnet101(pretrained)
        bl_exp = 4

        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AvgPool2d(int(opt.size / 32), stride=1)
        self.last_fc = nn.Linear(512 * bl_exp, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.last_fc(x)
        return x


class ClassifyNet():
    def __init__(self, num_classes, num_epochs, l_r, size, batch_size, n_workers, thresh, lbl_name, pretrained=True,
                 write_flag=False):
        # Hyper-parameters
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.learning_rate = l_r
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.thresh = thresh
        self.lbl_name = lbl_name
        self.write_flag = write_flag
        self.models_dir = "//homes//my_d//MODELS//"
        self.best_acc = 0.0

        dataset = isic.ISIC(load=False, size=(opt.size, opt.size),
                            transform=transforms.Compose([
                                transforms.Resize((self.size, self.size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180),
                                transforms.ToTensor(),
                                transforms.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351)),
                            ])
                            )

        eval_dataset = isic.ISIC(load=False, size=(opt.size, opt.size),
                                 transform=transforms.Compose([
                                     transforms.Resize((self.size, self.size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351)),
                                 ])
                                 )

        valid_dataset = task1_isic.ISIC(root='/homes/my_d/data/ISIC_dataset/Task_1/',  # split_list=training_set,
                                split_name='validation_2017',
                                load=opt.SRV, size=(self.size, self.size),
                                transform=transforms.Compose([
                                    transforms.Resize((self.size, self.size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351)),
                                ]),
                                target_transform=transforms.Compose([
                                    transforms.Resize((self.size, self.size)),
                                    transforms.ToTensor()
                                ])
                                )

        self.n = MyResnet(pretrained=pretrained, num_classes=self.num_classes).to('cuda')

        # Loss and optimizer
        if self.num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            training_stats = [0.1111333, 0.669495756, 0.051323015, 0.032651023, 0.109735397, 0.011482776, 0.014178732]
            # weights = np.subtract(1.0, training_stats, dtype='float32')
            # weights = np.divide(np.divide(1, training_stats), 100, dtype='float32')
            weights = np.divide(1, training_stats, dtype='float32')
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device='cuda'))
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.n.parameters()), lr=self.learning_rate)

        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.n_workers,
                                      drop_last=True,
                                      pin_memory=True)

        self.eval_data_loader = DataLoader(eval_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=True,
                                           pin_memory=True)

        self.valid_data_loader = DataLoader(valid_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=False,
                                           pin_memory=True)

        print(len(self.data_loader))

    def freeze_layers(self, freeze_flag=True, nl=0):
        if nl:
            l = list(self.n.resnet.named_children())[:-nl]
        else:
            l = list(self.n.resnet.named_children())
        # list(list(self.n.resnet.named_children())[0][1].parameters())[0].requires_grad
        for name, child in l:
            for param in child.parameters():
                param.requires_grad = not freeze_flag

    def train(self):
        for epoch in range(self.num_epochs):
            self.n.train()
            losses = []
            start_time = time.time()
            for i, (x, target, _) in enumerate(self.data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to('cuda')
                if self.num_classes == 1:
                    target = target.to('cuda', torch.float)
                else:
                    target = target.to('cuda', torch.long)
                output = torch.squeeze(self.n(x))
                loss = self.criterion(output, target)
                losses.append(loss.item())
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # # measure elapsed time
                # printProgressBar(i + 1, total + 1,
                #                  length=20,
                #                  prefix=f'Epoch {epoch} ',
                #                  suffix=f', loss: {loss.item()res = (check_output > 0.5).float() * 1:.3f}'
                #                  )

            print('Epoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)) + ' | time: ' + str(
                time.time() - start_time))

            if epoch % 5 == 4:
                self.save()

    def find_stats(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _, _ in self.data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntraining")
        print("mean: " + str(mean) + " | std: " + str(std))

    def explain_eval(self):

        sigm = nn.Sigmoid()
        sofmx = nn.Softmax(dim=1)
        trues = 0
        tr_trues = 0
        acc = 0
        self.n.eval()
        grad_cam = GradCam(self.n, target_layer_names=["7"], use_cuda=True)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None

        start_time = time.time()

        for i, (x, target, img_name) in enumerate(self.eval_data_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # compute output
            x = x.to('cuda')
            output = torch.squeeze(self.n(x))
            if self.num_classes == 1:
                target = target.to('cuda', torch.float)
                check_output = sigm(output)
                res = (check_output > self.thresh).float()
            else:
                target = target.to('cuda', torch.long)
                check_output = sofmx(output)
                check_output, res = torch.max(check_output, 1)

            tr_target = target * 2
            tr_target = tr_target - 1
            tr_trues += sum(res == tr_target).item()
            trues += sum(res).item()
            acc += sum(res == target).item()

            # gb_model = GuidedBackprop(self.n)
            for j in range(len(x)):
                in_im = Variable(x[j].unsqueeze(0), requires_grad=True)
                # mask = grad_cam(in_im, 0)
                # show_cam_on_image(nefro.denormalize(x[j]), mask, os.path.basename(img_name[j])[:-4] + self.lbl_name + '_cls0')
                mask = grad_cam(in_im, target_index)
                denorm_img = np.asarray(isic.denormalize(x[j].clone()))
                denorm_img = cv2.cvtColor(np.moveaxis(denorm_img, 0, -1), cv2.COLOR_RGB2BGR)
                show_cam_on_image(denorm_img, mask,
                                  os.path.basename(img_name[j])[:-4] + '_' + str(target[j].item()))
                cv2.imwrite('/homes/my_d/cvpr_GradCam/' + os.path.basename(img_name[j])[:-4] + '.png',
                            np.uint8(255 * denorm_img))

                # gb = gb_model.generate_gradients(in_im, target_index)
                # save_gradient_images(gb, '/homes/my_d/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                               :-4] + '_gb.png')
                # cam_gb = np.zeros(gb.shape)
                # if not np.isnan(mask).any():
                #     for c in range(0, gb.shape[0]):
                #         cam_gb[c, :, :] = mask
                #     cam_gb = np.multiply(cam_gb, gb)
                # save_gradient_images(cam_gb, '/homes/my_d/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                                   :-4] + '_cam_gb.png')

            # # measure elapsed time
            # printProgressBar(i + 1, total + 1,
            #                  length=20,
            #                  prefix=f'Epoch {epoch} ',
            #                  suffix=f', loss: {loss.item():.3f}'
            #                  )
        pr = tr_trues / (trues + 10e-5)
        rec = tr_trues / 375
        fscore = (2 * pr * rec) / (pr + rec + 10e-5)
        stats_string = 'Test set = Acc: ' + str(acc / 1000.0) + ' | F1 Score: ' + str(fscore) + ' | Precision: ' + str(
            pr) + ' | Recall: ' + str(rec) + ' | Trues: ' + str(trues) + ' | Correct Trues: ' + str(
            tr_trues) + ' | time: ' + str(time.time() - start_time)
        print(stats_string)

    def explain_validation(self):

        sigm = nn.Sigmoid()
        sofmx = nn.Softmax(dim=1)
        trues = 0
        tr_trues = 0
        acc = 0
        self.n.eval()
        grad_cam = GradCam(self.n, target_layer_names=["7"], use_cuda=True)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None

        start_time = time.time()

        for i, (x, target) in enumerate(self.valid_data_loader):
            # measure data loading time
            # print("data time: " + str(time.time() - start_time))

            # compute output
            x = x.to('cuda')
            output = torch.squeeze(self.n(x))
            if self.num_classes == 1:
                check_output = sigm(output)
                res = (check_output > self.thresh).float()
            else:
                check_output = sofmx(output)
                check_output, res = torch.max(check_output, 1)


            # gb_model = GuidedBackprop(self.n)
            for j in range(len(x)):
                in_im = Variable(x[j].unsqueeze(0), requires_grad=True)
                # mask = grad_cam(in_im, 0)
                # show_cam_on_image(nefro.denormalize(x[j]), mask, os.path.basename(img_name[j])[:-4] + self.lbl_name + '_cls0')
                mask = grad_cam(in_im, target_index)
                denorm_img = np.asarray(isic.denormalize(x[j].clone()))
                denorm_img = cv2.cvtColor(np.moveaxis(denorm_img, 0, -1), cv2.COLOR_RGB2BGR)
                show_cam_on_image(denorm_img, mask,
                                  str(i*self.batch_size + j) + '_class' + str(res[j].item()))
                cv2.imwrite('/homes/my_d/cvpr_GradCam/' + str(i*self.batch_size + j) + '_class' + str(res[j].item()) + '.png',
                            np.uint8(255 * denorm_img))

                # gb = gb_model.generate_gradients(in_im, target_index)
                # save_gradient_images(gb, '/homes/my_d/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                               :-4] + '_gb.png')
                # cam_gb = np.zeros(gb.shape)
                # if not np.isnan(mask).any():
                #     for c in range(0, gb.shape[0]):
                #         cam_gb[c, :, :] = mask
                #     cam_gb = np.multiply(cam_gb, gb)
                # save_gradient_images(cam_gb, '/homes/my_d/nefro_GradCam/' + os.path.basename(img_name[j])[
                #                                                                   :-4] + '_cam_gb.png')

            # # measure elapsed time
            # printProgressBar(i + 1, total + 1,
            #                  length=20,
            #                  prefix=f'Epoch {epoch} ',
            #                  suffix=f', loss: {loss.item():.3f}'
            #                  )



    def save(self):
        try:
            torch.save(self.n.state_dict(), os.path.join(self.models_dir, 'ISIC_Resnet_net.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.models_dir, 'ISIC_Resnet_opt.pth'))
            print("model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def load(self):
        self.n.load_state_dict(torch.load(os.path.join(self.models_dir, 'ISIC_Resnet_net.pth')))
        self.optimizer.load_state_dict(torch.load(os.path.join(self.models_dir, 'ISIC_Resnet_opt.pth')))
        print("model weights succesfully loaded")


def write_on_file(epoch, training_cost, validation_acc, t_validation_acc, tm, filename):
    ffname = files_path + filename
    with open(ffname, 'a+') as f:
        f.write("E:" + str(epoch) + " | Time: " + str(tm) +
                " | Training_acc: " + str(1 - training_cost) +
                " | validation_acc: " + str(1 - validation_acc) +
                " | t_validation_acc: " + str(t_validation_acc) + "\n")


def show_cam_on_image(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite('/homes/my_d/cvpr_GradCam/' + name + '_cam.png', np.uint8(255 * cam))


if __name__ == '__main__':
    # split_dataset_morelabels(['Par-regol-cont', 'Par-regol-discont', 'Par-irreg'], 'parietale')
    for i in range(opt.n_splits):
        # split_dataset()
        n = ClassifyNet(num_classes=opt.classes, num_epochs=opt.epochs, size=opt.size,
                        batch_size=opt.batch_size, thresh=opt.thresh, pretrained=(not opt.from_scratch),
                        l_r=opt.learning_rate, n_workers=opt.workers, lbl_name=opt.label, write_flag=False)
        # n.train()
        #
        # n.save()
        n.load()
        # n.explain_eval()
        n.explain_validation()
        # n.find_stats()
        # n.see_imgs()
        # n.write_testset()
