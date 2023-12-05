from torch import autograd
import torch
import torch.nn as nn
from networks.xception import TransferModel
from dataset.transform import xception_default_data_transforms
from PIL import Image as pil_image
import numpy as np
import cv2
from torch.autograd import Variable
import copy

def predict_with_model(preprocessed_image, post_function=nn.Softmax(dim=1), cuda=True):
    resized_image = nn.functional.interpolate(preprocessed_image, size=(256, 256), mode="bilinear",
                                                  align_corners=True)
    norm_transform = xception_default_data_transforms['normalize']
    normalized_image = norm_transform(resized_image)

    dropout = 0.5
    model = TransferModel('xception', dropout=dropout, return_fea=True)
    model = model.cuda()

    ret = model(normalized_image)
    logits, _, _, _ = ret
    output = post_function(logits)
    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output, logits


def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['to_tensor']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)

    if cuda:
        preprocessed_image = preprocessed_image.cuda()

    preprocessed_image.requires_grad = True
    return preprocessed_image

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def iterative_fgsm(input_img, mag, cuda=True, max_iter=10, desired_acc = 0.95):

    mag = mag.astype(np.float64)
    mag = float(mag)/3
    preprocessed_image = preprocess_image(input_img)

    input_var = autograd.Variable(preprocessed_image, requires_grad=True)
    target_var = autograd.Variable(torch.LongTensor([0]))

    if cuda:
        target_var = target_var.cuda()

    iter_no = 0
    print(iterative_fgsm)
    while iter_no < max_iter:
        prediction, output, logits = predict_with_model(input_var)
        if (output[0][0] - output[0][1]) > desired_acc:
            break

        loss_criterion = nn.CrossEntropyLoss()
        loss = loss_criterion(logits, target_var)
        if input_var.grad is not None:
            input_var.grad.data.zero_()
        loss.backward()

        step_adv = input_var.detach() - 1/255 * torch.sign(input_var.grad.detach())
        total_pert = step_adv - preprocessed_image
        total_pert = torch.clamp(total_pert, -mag, mag)

        input_adv = preprocessed_image + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)

        input_var.data = input_adv.detach()

        iter_no += 1

    input_var = torch.squeeze(input_var)
    input_var = input_var.permute((1, 2, 0))
    return input_var


def carlini_wagner_attack(input_img, cuda=True, max_attack_iter=50, alpha=0.005, const=1e-3, max_bs_iter=1, confidence=20.0):

    def torch_arctanh(x, eps=1e-6):
        x = x * (1. - eps)
        y = (1 + x) / (1 - x)
        z = torch.log(y)
        result = z * 0.5
        return result

    preprocessed_image = preprocess_image(input_img)
    attack_w = autograd.Variable(torch_arctanh(preprocessed_image.data - 1), requires_grad=True)
    bestl2 = 1e10
    bestscore = -1

    lower_bound_c = 0
    upper_bound_c = 1.0
    bestl2 = 1e10
    bestimg = None
    optimizer = torch.optim.Adam([attack_w], lr=alpha)
    print(carlini_wagner_attack)

    for bsi in range(max_bs_iter):
        for iter_no in range(max_attack_iter):
            adv_image = 0.5 * (torch.tanh(preprocessed_image + attack_w) + 1.)

            _, _, logits = predict_with_model(adv_image)

            loss1 = torch.clamp(logits[0][1] - logits[0][0] + confidence, min=0.0)
            loss2 = torch.norm(adv_image - preprocessed_image, 2)

            loss_total = loss2 + const * loss1
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()


        # binary search for const
        if (logits[0][0] - logits[0][1] > confidence):
            if loss2 < bestl2:
                bestl2 = loss2
                bestimg = adv_image.detach().clone().data

            upper_bound_c = min(upper_bound_c, const)
        else:
            lower_bound_c = max(lower_bound_c, const)

        const = (lower_bound_c + upper_bound_c) / 2.0

    if bestimg is not None:
        bestimg = torch.squeeze(bestimg)
        bestimg = bestimg.permute((1, 2, 0))
        return bestimg, perb
    else:
        adv_image = torch.squeeze(adv_image)
        adv_image = adv_image.permute((1, 2, 0))
        return adv_image

def deepfool(input_img, num_classes=2, overshoot=0.02, max_iter=50):
    preprocessed_image = preprocess_image(input_img)
    _, _, logits = predict_with_model(preprocessed_image)
    logits = logits.data.cpu().numpy().flatten()

    I = (np.array(logits)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]

    input_shape = input_img.shape
    pert_image = copy.deepcopy(preprocessed_image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    r_tot = r_tot.transpose(2, 0, 1)

    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    _, _, fs = predict_with_model(x)

    k_i = label
    print(deepfool)

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            if pert_k < pert:
                pert = pert_k
                w = w_k


        r_i = (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        preprocessed_image = torch.squeeze(preprocessed_image)
        pert_image = preprocessed_image + (1+overshoot)*torch.from_numpy(r_tot).cuda()

        x = Variable(pert_image, requires_grad=True)
        _, _, fs = predict_with_model(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    pert_image = torch.squeeze(pert_image)
    pert_image = pert_image.permute((1, 2, 0))
    return pert_image


def black_box_attack(input_img, mag, cuda=True, max_iter=10, desired_acc=0.90):

    def _get_transforms():

        transform_list = [
            lambda x: x,
        ]

        return transform_list

    def _find_nes_gradient(input_var, num_samples=20, sigma=0.001):
        g = 0
        _num_queries = 0
        for sample_no in range(num_samples):
            rand_noise = torch.randn_like(input_var)
            img1 = input_var + sigma * rand_noise
            img2 = input_var - sigma * rand_noise

            prediction1, probs_1, _ = predict_with_model(img1, cuda=cuda)
            prediction2, probs_2, _ = predict_with_model(img2, cuda=cuda)

            _num_queries += 2
            g = g + probs_1[0][0] * rand_noise
            g = g - probs_2[0][0] * rand_noise
            g = g.data.detach()

            del rand_noise
            del img1
            del prediction1, probs_1
            del prediction2, probs_2

        return (1./(2. * num_samples * sigma)) * g, _num_queries

    mag = mag.astype(np.float64)
    mag = float(mag)/3

    preprocessed_image = preprocess_image(input_img)

    input_var = autograd.Variable(preprocessed_image, requires_grad=True)

    target_var = autograd.Variable(torch.LongTensor([0]))
    if cuda:
        target_var = target_var.cuda()

    iter_no = 0

    warm_start_done = False
    num_queries = 0
    print(black_box_attack)

    while iter_no < max_iter:

        if not warm_start_done:
            _, output, _ = predict_with_model(input_var)
            num_queries += 1
            if output[0][0] > desired_acc:
                warm_start_done = True

        _, output, _ = predict_with_model(input_var)
        num_queries += 1
        if output[0][0] > desired_acc:
            break

        step_gradient_estimate, _num_grad_calc_queries = _find_nes_gradient(input_var)
        num_queries += _num_grad_calc_queries
        step_adv = input_var.detach() + 1/255.0 * torch.sign(step_gradient_estimate.data.detach())
        total_pert = step_adv - preprocessed_image
        total_pert = torch.clamp(total_pert, -mag, mag)
        
        input_adv = preprocessed_image + total_pert
        input_adv = torch.clamp(input_adv, 0, 1)
        
        input_var.data = input_adv.detach()
        iter_no += 1

    input_var = torch.squeeze(input_var)
    input_var = input_var.permute((1, 2, 0))
    return input_var
