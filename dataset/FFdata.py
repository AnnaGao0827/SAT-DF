import random
from random import randrange
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import dlib
detector = dlib.get_frontal_face_detector()

def load_rgb(file_path, size=256):
    img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class FaceForensicsDataset(data.Dataset):
    data_root = './data'
    frames = {'test': 110, 'eval': 110, 'train': 270}

    def __init__(self, dataset='FF-DF', mode='test', res=256, train=True,
                 sample_num=None):
        self.mode = mode
        self.dataset = dataset
        img_lines = []
        fake_lines = []
        real_lines = []
        if mode == "train":
            video_list = [i for i in range(600)]
            random.shuffle(video_list)
            for j in range(600):
                for i in range(0, self.frames[mode]):
                    real_lines.append(('{}/{:03d}'.format('real', j), i, 0))
            for fake_d in ['FF-DF', 'FF-NT', 'FF-FS', 'FF-F2F']:
                for j in video_list:
                    for i in range(0, self.frames[mode]):
                        fake_lines.append(('{}/{:03d}'.format(fake_d, j), i, 1))
        else:
            video_list = [i for i in range(600, 800)]
            random.shuffle(video_list)
            for j in range(600, 800):
                for i in range(0, self.frames[mode]):
                    real_lines.append(('{}/{:03d}'.format('real', j), i, 0))
            for fake_d in ['FF-DF', 'FF-NT', 'FF-FS', 'FF-F2F']:
                for j in video_list:
                    for i in range(0, self.frames[mode]):
                        fake_lines.append(('{}/{:03d}'.format(fake_d, j), i, 1))

        self.fake_lines = fake_lines
        self.img_lines = self.fake_lines
        self.real_lines = np.random.permutation(real_lines)
        for i in self.real_lines:
            self.img_lines.insert(randrange(len(self.img_lines) + 1), i)

        if sample_num is not None:
            self.img_lines = img_lines[:sample_num]

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

        self.totensor = T.Compose([T.ToTensor()])
        self.res = res

    def load_image(self, name, idx):
        impath = '{}/{}/{:04d}.png'.format(self.data_root, name, int(idx)+1)
        img = load_rgb(impath, size=self.res)
        return img

    def __getitem__(self, index):
        name, idx, label = self.img_lines[index]
        video = int(name[-3:])
        label = int(label)
        img = self.load_image(name, idx)

        img = Image.fromarray(np.array(img, dtype=np.uint8))
        return self.transforms(img), label, video

    def __len__(self):
        return len(self.img_lines)

