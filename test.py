import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from dataset.FFdata import FaceForensicsDataset
from whole_model import Model
from metrics import Metrics
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()

parser.add_argument('--meta', default="FF-DF",
                    type=str, help='the feature space')
parser.add_argument('--resolution', type=int, default=256,
                    help='the resolution of the output image to network')
parser.add_argument('--test_batchSize', type=int,
                    default=32, help='test batch size')
parser.add_argument("--pretrained", default=None, type=str,
                    help="path to pretrained model (default: none)")
parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('-mp', '--masterport', default='5555', type=str,
                    help='ranking within the nodes')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--cuda',default=True, action='store_true', help='enable cuda')

def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    datas = torch.cat((prob, label.float()), dim=1)
    return datas

def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    return accuracy

def dataload(gpu,args):
    model = Model(args, train=False)
    torch.cuda.set_device(gpu)
    model.model.cuda(gpu)

    if args.pretrained is not None:
        model.load_ckpt(args.pretrained, 0)

    TESTLIST_ff = {
        'FF-DF': "non-input",
        'FF-NT': "non-input",
        'FF-FS': "non-input",
        'FF-F2F': "non-input",
    }

    def get_data_loader_ff(name):
        # -----------------load dataset--------------------------
        test_set = FaceForensicsDataset(mode='test', res=args.resolution, train=False)
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize,
                                                        shuffle=False, num_workers=0)
        return test_data_loader

    test_data_loaders_ff = {}
    for list_key in TESTLIST_ff.keys():
        test_data_loaders_ff[list_key] = get_data_loader_ff(list_key)
    test(model, test_data_loaders_ff)


def test(model, test_data_loaders_ff):
    model.setEval()
    def run_ff(data_loader, name):
        statistic = None
        metric = Metrics()
        losses = []
        acces = []

        for i, batch in enumerate(data_loader):
            data, label, video = batch
            if isinstance(data, list):
                data = data[0]
            img = data
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            cls_score, loss = model.inference(img, label)

            tmp_data = get_prediction(cls_score.detach(), label)
            if statistic is not None:
                statistic = torch.cat((statistic, tmp_data), dim=0)
            else:
                statistic = tmp_data

            losses.append(loss.cpu().detach().numpy())
            acces.append(get_accracy(cls_score, label))
            metric.update(label.detach(), cls_score.detach())

        avg_loss = np.mean(np.array(losses))
        info = "|Test Loss {:.4f}".format(avg_loss)
        mm = metric.get_mean_metrics()
        mm_str = ""
        mm_str += "\t|Acc {:.4f} (~{:.2f})".format(mm[0], mm[1])
        mm_str += "\t|AUC {:.4f} (~{:.2f})".format(mm[2], mm[3])
        mm_str += "\t|EER {:.4f} (~{:.2f})".format(mm[4], mm[5])
        mm_str += "\t|AP {:.4f} (~{:.2f})".format(mm[6], mm[7])
        info += mm_str
        print(info)
        metric.clear()

        return (mm[0], mm[2], mm[4], mm[6])

    keys = test_data_loaders_ff.keys()
    datas = [{}, {}, {}, {}]
    for i, key in enumerate(keys):
        print('Testing from {} ...'.format(key))
        dataloader = test_data_loaders_ff[key]
        ret = run_ff(dataloader, key)
        for j, data in enumerate(ret):
            datas[j][key] = data

def main():
    opt = parser.parse_args()        #
    mp.spawn(dataload, nprocs=opt.gpus, args=(opt,))        #

if __name__ == '__main__':
    main()