#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable

### custom lib
from networks.resample2d_package.modules.resample2d import Resample2d
import networks
import utils

### custom lib
from WCT2 import *

def equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency - WCT Pytorch')

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
    parser.add_argument('--style', type=str, default="style/kandinsky.jpeg", help="path to style image")
    
    args = parser.parse_args()
    args.ccuda = True

    args.size_multiplier = 2 ** 2 ## Inputs to TransformNet need to be divided by 4

    if args.ccuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")


    ### load model opts
    opts_filename = "pretrained_models/ECCV18_blind_consistency_opts.pth"
    print("Load %s" %opts_filename)
    with open(opts_filename, 'rb') as f:
        print(f)
        model_opts = pickle.load(f)


    ### initialize model
    print('===> Initializing model from %s...' %model_opts.model)
    model = networks.__dict__[model_opts.model](model_opts, nc_in=12, nc_out=3)


    ### load trained model
    model_filename = "pretrained_models/ECCV18_blind_consistency.pth"
    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict['model'])

    ### convert to GPU
    device = torch.device("cuda" if args.ccuda else "cpu")
    model = model.to(device)

    model.eval()
            
    ST = StyleTransfer(args)
    ST.uploadStyleImage(args.style, resize=True)
    cap = cv2.VideoCapture(0)

    # initial
    ret, frame = cap.read()
    frame_i1 = int2float(np.float32(frame))
    #frame_o1 = equalization(frame)
    frame_o1 = ST.eval(frame)

    lstm_state = None

    t1 = time.time()

    while True:

        # next
        ret, frame = cap.read()
        frame_i2 = int2float(np.float32(frame))
        #frame_p2 = equalization(frame)
        frame_p2 = ST.eval(frame)
        
        ### resize image
        H_orig = frame_p2.shape[0]
        W_orig = frame_p2.shape[1]

        H_sc = int(math.ceil(float(H_orig) / args.size_multiplier) * args.size_multiplier)
        W_sc = int(math.ceil(float(W_orig) / args.size_multiplier) * args.size_multiplier)

        frame_i1 = cv2.resize(frame_i1, (W_sc, H_sc))
        frame_i2 = cv2.resize(frame_i2, (W_sc, H_sc))
        frame_o1 = cv2.resize(frame_o1, (W_sc, H_sc))
        frame_p2 = cv2.resize(frame_p2, (W_sc, H_sc))
        
        with torch.no_grad():

            ### convert to tensor
            frame_i1 = utils.img2tensor(frame_i1).to(device)
            frame_i2 = utils.img2tensor(frame_i2).to(device)
            frame_o1 = utils.img2tensor(frame_o1).to(device)
            frame_p2 = utils.img2tensor(frame_p2).to(device)
            
            ### model input
            inputs = torch.cat((frame_p2, frame_o1, frame_i2, frame_i1), dim=1)
            
            output, lstm_state = model(inputs, lstm_state)
            frame_o2 = frame_p2 + output
           
            ## create new variable to detach from graph and avoid memory accumulation
            lstm_state = utils.repackage_hidden(lstm_state) 


        ### convert to numpy array
        frame_o2 = utils.tensor2img(frame_o2)
        
        ### resize to original size
        frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))

        frame_i2_numpy = frame_i2.squeeze().transpose(0, 2).transpose(0, 1).cpu().numpy()
        frame_p2_numpy = frame_p2.squeeze().transpose(0, 2).transpose(0, 1).cpu().numpy()

        stacked = np.hstack((frame_i2_numpy, frame_p2_numpy, frame_o2))

        cv2.imshow('frame', stacked)

        frame_i1 = frame_i2_numpy
        frame_o1 = frame_o2

        t2 = time.time()

        print("{:.2f} fps".format(1/(t2 - t1)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        t1 = t2

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()