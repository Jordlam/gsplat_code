"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from threestudio.utils.perceptual.utils import get_ckpt_path


class PerceptualLoss(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "threestudio/utils/lpips")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model

    def forward(self, input, target):
        if input.isnan().any() or input.isinf().any():
            print("Bad input")
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        if in0_input.isnan().any() or in0_input.isinf().any():
            print("Bad in0_input")
        if in1_input.isnan().any() or in1_input.isinf().any():
            print("Bad in1_input")
        # Error here right now
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            f0kk, f1kk = feats0[kk], feats1[kk]
            # assert f0kk.requires_grad, "f0kk is detached"
            # assert f1kk.requires_grad, "f1kk is detached"
            # torch.autograd.gradcheck(lambda x: (x - feats1[kk]) ** 2, feats0[kk])
            # if torch.is_tensor(f0kk) and f0kk.isnan().any().item():
            #     print("f0kk", f0kk)
            # if torch.is_tensor(f1kk) and f1kk.isnan().any().item():
            #     print("f1kk", f1kk)
            # print((f0kk < 0).any(), "Negative f0kk")
            # print((f1kk < 0).any(), "Negative f1kk")
            # print(((f0kk - f1kk) < 0).any(), "Negative f1kk")
            eps = 1e-5
            diff = f0kk - f1kk
            diffs[kk] = diff.clamp(min=eps).abs().square()

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        do_clamp = lambda X: X.clamp(min=1e-5)
        if X.isnan().any() or X.isinf().any():
            print("Bad X")
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        # Original code
        # h = self.slice1(X)
        # h_relu1_2 = h
        # h = self.slice2(h)
        # h_relu2_2 = h
        # h = self.slice3(h)
        # h_relu3_3 = h
        # h = self.slice4(h)
        # h_relu4_3 = h

        if h.isnan().any() or h.isinf().any():
            print("Bad h")
        h = self.slice5(h)
        # h = self.slice5(do_clamp(h))
        if h.isnan().any() or h.isinf().any():
            print("Bad h_2")
        # h = do_clamp(h)
        if h.isnan().any() or h.isinf().any():
            print("Bad h_clamp")
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )

        # Clamp all before return
        h_relu1_2 = do_clamp(h_relu1_2)
        h_relu2_2 = do_clamp(h_relu2_2)
        h_relu3_3 = do_clamp(h_relu3_3)
        h_relu4_3 = do_clamp(h_relu4_3)
        h_relu5_3 = do_clamp(h_relu5_3)

        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

def normalize_tensor(x, eps=1e-5):
    # Original Code
    # norm_factor = torcvidiaih.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    # return x / (norm_factor + eps)

    # return nn.functional.normalize(x, eps=eps)
    if x.isnan().any() or x.isinf().any():
        print("Bad x")
    x_2 = x.clamp(min=eps).abs().square()
    norm_factor = torch.sqrt(torch.sum(x_2, dim=1, keepdim=True))
    norm_factor = norm_factor.clamp(min=eps)
    return (x.clamp(min=eps) / norm_factor).clamp(min=eps)

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
