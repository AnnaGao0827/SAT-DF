from torchvision import transforms


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        for i, m, s in zip(range(tensor.size(0)), self.mean, self.std):
            t = tensor[i]
            t.sub_(m).div_(s)
        return tensor

xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),

    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize' : transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}