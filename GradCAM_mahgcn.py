import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import sys
import numpy as np
import argparse

import torch.nn.functional as F


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, g1,g2,g3,g4,g5):
        features = []
        outputs = []
        self.gradients = []

        #outputs=self.model(g1, g2, g3,g4,g5)
        x = torch.zeros(g1.shape[0], 500, 500)
        for s in range(g1.shape[0]):
            x[s, :, :] = torch.eye(500)
        x = x.cuda()
        newx=torch.zeros(g1.shape[0], 1500)
        for name, module in self.model._modules.items():
            #print(name)
            if name == 'mgunet':
                for i in range(g5.shape[0]):
                    for subname, submodule in self.model.mgunet._modules.items():
                        if subname=='net_gcn_down1':
                            temp = submodule(g5[i,:,:].cuda(), x[i,:,:].cuda())
                            down1=temp
                        elif subname=='net_gcn_down2':
                            temp = submodule(g4[i,:,:].cuda(), torch.diag(temp).cuda())
                            down2 = temp
                        elif subname == 'net_gcn_down3':
                            temp = submodule(g3[i,:,:].cuda(), torch.diag(temp).cuda())
                            down3 = temp
                        elif subname == 'net_gcn_down4':
                            temp = submodule(g2[i,:,:].cuda(), torch.diag(temp).cuda())
                            down4 = temp
                        elif subname == 'net_gcn_bot':
                            temp = submodule(g1[i,:,:].cuda(), torch.diag(temp).cuda())
                            newx[i,:]=torch.squeeze(torch.cat((temp, down1, down2, down3, down4)))
                        else:
                            temp = submodule(temp.cuda())
                        if subname in self.target_layers:
                            #print('hook')
                            temp.register_hook(self.save_gradient)
                            #print(self.gradients)
                            features += [temp]
            else:
                newx = module(newx.cuda())
        return features, newx


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, g1,g2,g3,g4,g5):
        target_activations, output = self.feature_extractor(g1,g2,g3,g4,g5)
        output = output.view(output.size(0), -1)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input



class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, g1,g2,g3,g4,g5):
        return self.model(g1,g2,g3,g4,g5)

    def __call__(self, g1,g2,g3,g4,g5, index=None):
        if self.cuda:
            features, output = self.extractor(g1.cuda(),g2.cuda(),g3.cuda(),g4.cuda(),g5.cuda())
        else:
            features, output = self.extractor(g1,g2,g3,g4,g5)
        #print(output.shape)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        #print(index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()

        weights = grads_val
        cam = weights * target

        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_variables=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

