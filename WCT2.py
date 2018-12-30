import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
import time
import cv2


class StyleTransfer():
    def __init__(self, args):
        self.args = args
        self.wct = WCT(self.args)
        self.cImg = torch.Tensor()
        self.sImg = torch.Tensor()
        self.csF = torch.Tensor()
        self.csF = Variable(self.csF)
        if(self.args.cuda):
            self.cImg = self.cImg.cuda(self.args.gpu)
            self.sImg = self.sImg.cuda(self.args.gpu)
            self.csF = self.csF.cuda(self.args.gpu)
            self.wct.cuda(self.args.gpu)


    def styleTransfer(self, contentImg):
        """
        Transform a content image.
        Inputs:
            contentImg: (tensor) (1 x channels x width x height)
        Outpts:
            im: (tensor) (1 x channels x width x height)
        """
        sF5 = self.wct.e5(self.sImg)
        cF5 = self.wct.e5(contentImg)
        sF5 = sF5.data.cpu().squeeze(0)
        cF5 = cF5.data.cpu().squeeze(0)
        csF5 = self.wct.transform(cF5, sF5, self.csF, self.args.alpha)
        Im5 = self.wct.d5(csF5)

        sF4 = self.wct.e4(self.sImg)
        cF4 = self.wct.e4(Im5)
        sF4 = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        csF4 = self.wct.transform(cF4, sF4, self.csF, self.args.alpha)
        Im4 = self.wct.d4(csF4)

        sF3 = self.wct.e3(self.sImg)
        cF3 = self.wct.e3(Im4)
        sF3 = sF3.data.cpu().squeeze(0)
        cF3 = cF3.data.cpu().squeeze(0)
        csF3 = self.wct.transform(cF3, sF3, self.csF, self.args.alpha)
        Im3 = self.wct.d3(csF3)

        sF2 = self.wct.e2(self.sImg)
        cF2 = self.wct.e2(Im3)
        sF2 = sF2.data.cpu().squeeze(0)
        cF2 = cF2.data.cpu().squeeze(0)
        csF2 = self.wct.transform(cF2, sF2, self.csF, self.args.alpha)
        Im2 = self.wct.d2(csF2)

        sF1 = self.wct.e1(self.sImg)
        cF1 = self.wct.e1(Im2)
        sF1 = sF1.data.cpu().squeeze(0)
        cF1 = cF1.data.cpu().squeeze(0)
        csF1 = self.wct.transform(cF1, sF1, self.csF, self.args.alpha)
        Im1 = self.wct.d1(csF1)
        # save_image has this wired design to pad images with 4 pixels at default.
        im = normalize(Im1)
        im = im.data.float()
        return im


    def eval(self, conImg):
        """
        Eval image
        Inputs:
            conImg: numpy array (h x w h nchannels)
        Outputs:
            proImg: numpy array (h x w x nchannels)
        """
        conImg = int2float(np.float32(conImg))
        # convert to tensor
        conImg = torch.Tensor(conImg).unsqueeze(0).transpose(1, 3).cuda(self.args.gpu)
        cImg = Variable(conImg, volatile=True)
        proImg = self.styleTransfer(cImg)
        proImg = proImg.cpu().squeeze().transpose(0, 2).numpy() 
        #proImg = float2int(proImg)
        return proImg


    def uploadStyleImage(self, image_path, resize=False):
        self.styleImg = cv2.imread(image_path, 1)
        if resize:
            self.styleImg = cv2.resize(self.styleImg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        self.styleImg = int2float(self.styleImg)
        self.styleImg = torch.Tensor(self.styleImg).unsqueeze(0).transpose(1, 3).cuda(self.args.gpu)
        self.sImg = Variable(self.styleImg, volatile=True)



def normalize(x):
    minx = torch.min(x)
    return (x - minx)/(torch.max(x) - minx)


def int2float(x):
    return x/255.


def float2int(x):
    return (x*255.).astype(np.uint8)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='WCT Pytorch')
    parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--vgg1', default='/home/mauricio/Documents/PytorchWCT/models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
    parser.add_argument('--vgg2', default='/home/mauricio/Documents/PytorchWCT/models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
    parser.add_argument('--vgg3', default='/home/mauricio/Documents/PytorchWCT/models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
    parser.add_argument('--vgg4', default='/home/mauricio/Documents/PytorchWCT/models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
    parser.add_argument('--vgg5', default='/home/mauricio/Documents/PytorchWCT/models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
    parser.add_argument('--decoder5', default='/home/mauricio/Documents/PytorchWCT/models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
    parser.add_argument('--decoder4', default='/home/mauricio/Documents/PytorchWCT/models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
    parser.add_argument('--decoder3', default='/home/mauricio/Documents/PytorchWCT/models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
    parser.add_argument('--decoder2', default='/home/mauricio/Documents/PytorchWCT/models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
    parser.add_argument('--decoder1', default='/home/mauricio/Documents/PytorchWCT/models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
    parser.add_argument('--alpha', type=float,default=0.5, help='hyperparameter to blend wct feature and content feature')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

    args = parser.parse_args()

    ST = StyleTransfer(args)
    ST.uploadStyleImage("/home/mauricio/Documents/PytorchWCT/images/style/in3.jpg")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        pro_img = ST.eval(frame)
        cv2.imshow('frame', pro_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
