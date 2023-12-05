import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from networks.xception import TransferModel
from networks.generator import generator
from networks.am_softmax import AMSoftmaxLoss
from collections import OrderedDict

model_name = 'whole-model'

class Model():

    def __init__(self, opt, logdir=None, train=True):
        if opt is not None:
            self.meta = opt.meta
            self.opt = opt
            self.ngpu = opt.ngpu

        self.writer = None
        self.logdir = logdir
        dropout = 0.5
        self.model = TransferModel('xception', dropout=dropout, return_fea=True)
        self.generator = generator()
        self.cls_criterion = AMSoftmaxLoss(gamma=0., m=0.45, s=30, t=1.)
        self.train = train
        self.l1loss = nn.MSELoss()
        params = ([p for p in self.model.parameters()])
        params_generator = ([p for p in self.generator.parameters()])
        if train:
            self.optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)
            self.optimizer_generator = optim.Adam(params_generator, lr=opt.lr/4, betas=(opt.beta1, 0.999),
                                        weight_decay=opt.weight_decay)


    def define_summary_writer(self):
        if self.logdir is not None:
            # tensor board writer
            timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            log = '{}/{}/{}'.format(self.logdir, model_name, self.meta)
            log = log + '_{}'.format(timenow)
            print('TensorBoard log dir: {}'.format(log))

            self.writer = SummaryWriter(log_dir=log)


    def setTrain(self):
        self.model.train()
        self.generator.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path=None, generator_path=0):
        if model_path !=0 and os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            module_lst = [i for i in self.model.state_dict()]
            weights = OrderedDict()
            for idx, (k, v) in enumerate(saved.items()):
                if self.model.state_dict()[module_lst[idx]].numel() == v.numel():
                    weights[module_lst[idx]] = v
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(weights, strict=False)
            print('Discriminator found in {}'.format(model_path))

        if generator_path != 0 and os.path.isfile(generator_path):
            saved = torch.load(generator_path, map_location='cpu')
            module_lst = [i for i in self.model.state_dict()]
            weights = OrderedDict()
            for idx, (k, v) in enumerate(saved.items()):
                if self.model.state_dict()[module_lst[idx]].numel() == v.numel():
                    weights[module_lst[idx]] = v
            suffix = generator_path.split('.')[-1]
            if suffix == 'p':
                self.generator.load_state_dict(saved.state_dict())
            else:
                self.generator.load_state_dict(weights, strict=False)
            print('Generator found in {}'.format(generator_path))

    def save_ckpt(self, dataset, epoch, iters, save_dir, best=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        mid_dir = os.path.join(save_dir, model_name)
        if not os.path.exists(mid_dir):
            os.mkdir(mid_dir)

        sub_dir = os.path.join(mid_dir, self.meta)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        subsub_dir = os.path.join(sub_dir, dataset)
        if not os.path.exists(subsub_dir):
            os.mkdir(subsub_dir)

        if best:
            ckpt_name = "epoch_{}_iter_{}_best.pth".format(epoch, iters)
            ckpt_name2 = "epoch_{}_iter_{}_best_syn.pth".format(epoch, iters)
        else:
            ckpt_name = "epoch_{}_iter_{}.pth".format(epoch, iters)
            ckpt_name2 = "epoch_{}_iter_{}_syn.pth".format(epoch, iters)

        save_path = os.path.join(subsub_dir, ckpt_name)
        save_path_ctrl = os.path.join(subsub_dir, ckpt_name2)

        if self.ngpu > 1:
            torch.save(self.model.module.state_dict(), save_path)
            torch.save(self.generator.module.state_dict(), save_path_ctrl)
        else:
            torch.save(self.model.state_dict(), save_path)
            torch.save(self.generator.state_dict(), save_path_ctrl)

        print("Checkpoint saved to {}".format(save_path))

    def optimize(self, img, label, video, epoch):
        device = torch.device("cuda")
        img = img.to(device)
        log_prob, entropy, new_img, label, type_label, mag_label = \
            self.generator(img, label, video)
        new_img = new_img.to(device)
        label = label.to(device)
        type_label = type_label.to(device)
        mag_label = mag_label.to(device)

        img_flip = torch.flip(new_img, (3,)).detach().clone()
        new_img = torch.cat((new_img, img_flip))

        label = torch.cat((label, label))
        type_label = torch.cat((type_label, type_label))
        mag_label = torch.cat((mag_label, mag_label))

        ret = self.model(new_img)

        score, fea, type, mag = ret

        if fea is not None:
            del(fea)
        loss_cls = self.cls_criterion(score, label).mean()
        loss_type = self.cls_criterion(type, type_label).mean()
        loss_mag = self.l1loss(mag, mag_label).mean()

        loss = loss_cls + 0.05*loss_type + 0.05*loss_mag

        if self.train:
            self.optimizer_generator.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if log_prob is not None:
                lm = loss.detach()
                normlized_lm = lm
                score_loss = torch.mean(-log_prob * normlized_lm)
                entropy_penalty = torch.mean(entropy)
                generator_loss = score_loss - (1e-5) * entropy_penalty
                generator_loss.backward()
                self.optimizer_generator.step()

        return label, (score, loss)

    def inference(self, img, label):
        with torch.no_grad():
            ret = self.model(img)
            score, fea, type, mag = ret
            loss_cls = self.cls_criterion(score, label).mean()
            return score, loss_cls

    def update_tensorboard(self, loss, step, acc=None, datas=None, name='train'):
        assert self.writer
        if loss is not None:
            loss_dic = {'Cls': loss}
            self.writer.add_scalars('{}/Loss'.format(name), tag_scalar_dict=loss_dic,
                                    global_step=step)

        if acc is not None:
            self.writer.add_scalar('{}/Acc'.format(name), acc, global_step=step)

        if datas is not None:
            self.writer.add_pr_curve(name, labels=datas[:, 1].long(),
                                     predictions=datas[:, 0], global_step=step)

    def update_tensorboard_test_accs(self, accs, step, feas=None, label=None, name='test'):
        assert self.writer
        if isinstance(accs, list):
            self.writer.add_scalars('{}/ACC'.format(name),
                                    tag_scalar_dict=accs[0], global_step=step)
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs[1], global_step=step)
            self.writer.add_scalars('{}/EER'.format(name),
                                    tag_scalar_dict=accs[2], global_step=step)
            self.writer.add_scalars('{}/AP'.format(name),
                                    tag_scalar_dict=accs[3], global_step=step)
        else:
            self.writer.add_scalars('{}/AUC'.format(name),
                                    tag_scalar_dict=accs, global_step=step)

        if feas is not None:
            metadata = []
            mat = None
            for key in feas:
                for i in range(feas[key].size(0)):
                    lab = 'fake' if label[key][i] == 1 else 'real'
                    metadata.append('{}_{:02d}_{}'.format(key, int(i), lab))
                if mat is None:
                    mat = feas[key]
                else:
                    mat = torch.cat((mat, feas[key]), dim=0)

            self.writer.add_embedding(mat, metadata=metadata, label_img=None,
                                      global_step=step, tag='default')
