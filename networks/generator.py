import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
from networks.xception_forgen import TransferModel
from attack_algos import iterative_fgsm, black_box_attack, carlini_wagner_attack, deepfool


def attack_fake(fakeimg, type, mag):
    fakeimg = ((fakeimg + 1) / 2 * 255).astype(np.uint8)
    if type == 0:
        out_attack = iterative_fgsm(fakeimg, mag, cuda=True)
    elif type == 1:
        out_attack = carlini_wagner_attack(fakeimg, cuda=True)
    elif type == 2:
        out_attack = deepfool(fakeimg)
    elif type == 3:
        out_attack = black_box_attack(fakeimg, mag, cuda=True)
    return out_attack


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.netG = TransferModel('xception', num_type=5, num_mag=1, inc=6)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])

    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
   
    def calculate(self, logits):
        if logits.shape[1] != 1:
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, self.get_variable(action, requires_grad=False))
        else:
            probs = torch.sigmoid(logits)
            log_prob = torch.log(torch.sigmoid(logits))
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs
            selected_log_prob = log_prob
        return entropy, selected_log_prob[:, 0], action[:, 0]

    def forward(self, img, label, video):
        type_num, mag = self.netG(torch.cat((img, img), 1))
        type_etp, type_log_prob, type = self.calculate(type_num)
        mag_etp, mag_log_prob, mag = self.calculate(mag)
        entropy = type_etp + mag_etp
        log_prob = type_log_prob + mag_log_prob
        newlabel = []
        typelabel = []
        alt_img = torch.ones(img.shape)
        judge = False

        for i in range(img.shape[0]):
            imgcp = np.transpose(img[i].cpu().numpy(), (1,2,0)).copy()
            if label[i] == 1 and type[i] != 4:
                if judge and video[i]==video[i-1] and type[i] != 3:
                    newimg = imgcp + pert
                    newimg = self.transforms(Image.fromarray(np.array((newimg + 1) / 2 * 255, dtype=np.uint8)))
                else:
                    newimg = attack_fake(imgcp, type[i], mag[i].detach().cpu().numpy())
                    newimg = newimg.cpu().detach().numpy()
                    pert = newimg - imgcp
                    judge = True
                    newimg = self.transforms(Image.fromarray(np.array(newimg, dtype=np.uint8)))
                newlabel.append(int(1))
                typelabel.append(int(type[i].cpu().numpy()))
            else:
                newimg = self.transforms(Image.fromarray(np.array((imgcp+1)/2 * 255, dtype=np.uint8)))
                newlabel.append(int(label[i].cpu().numpy()))
                if label[i] == 1:
                    typelabel.append(int(4))
                else:
                    typelabel.append(int(5))
            alt_img[i] = newimg

        newlabel = torch.tensor(newlabel)
        typelabel = torch.tensor(typelabel)
        maglabel = mag
        return log_prob, entropy, alt_img.detach(), \
               newlabel.detach(), typelabel.detach(), maglabel.detach()


