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
    parser.add_argument('--vgg1', default='models/models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
    parser.add_argument('--vgg2', default='models/models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
    parser.add_argument('--vgg3', default='models/models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
    parser.add_argument('--vgg4', default='models/models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
    parser.add_argument('--vgg5', default='models/models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
    parser.add_argument('--decoder5', default='models/models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
    parser.add_argument('--decoder4', default='models/models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
    parser.add_argument('--decoder3', default='models/models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
    parser.add_argument('--decoder2', default='models/models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
    parser.add_argument('--decoder1', default='models/models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
    parser.add_argument('--alpha', type=float,default=0.5, help='hyperparameter to blend wct feature and content feature')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
    parser.add_argument('--style', type=str, default="style/kandinsky.jpg", help="path to style image")
    parser.add_argument('--video', type=str, default="test.mp4", help="path to video")
    
    args = parser.parse_args()
    args.ccuda = True

    args.size_multiplier = 2 ** 2 ## Inputs to TransformNet need to be divided by 4

    video_name = args.video.split("/")[-1].split(".")[0]

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
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # initial
    ret, frame = cap.read()
    frame_i1 = int2float(np.float32(frame))
    #frame_o1 = equalization(frame)
    frame_o1 = ST.eval(frame)
    frame_p2 = ST.eval(np.float32(frame))

    lstm_state = None

    font = cv2.FONT_HERSHEY_SIMPLEX

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width, _ = frame_p2.shape
    video = cv2.VideoWriter("processed_videos/{}_processed.avi".format(video_name), fourcc, fps, (int(width*3), height))

    for i in range(nframes):
        print("frame {}/{}".format(i, nframes))

        # next
        ret, frame = cap.read()
        if ret==True:
            frame_i2 = int2float(np.float32(frame))
            #frame_p2 = equalization(frame)
            frame_p2 = ST.eval(np.float32(frame))
            
            ### resize image
            H_orig, W_orig, _ = frame_p2.shape

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

            frame_i2_with_fps = np.copy(frame_i2_numpy)
            frame_o2_with_fps = np.copy(frame_o2)

            cv2.rectangle(frame_i2_with_fps, (10, H_sc - 25), (10 + 125, H_sc), (0, 0, 0), -1)
            cv2.putText(frame_i2_with_fps, "original", (10 + 150 + 10, H_sc), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame_p2_numpy, (10, H_sc - 25), (10 + 225, H_sc), (0, 0, 0), -1)
            cv2.putText(frame_p2_numpy, "style transfer", (10, H_sc), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(frame_o2_with_fps, (10, H_sc - 25), (10 + 375, H_sc), (0, 0, 0), -1)
            cv2.putText(frame_o2_with_fps, "blind video consistency", (10, H_sc), font, 1, (255, 255, 255), 2, cv2.LINE_AA)        

            stacked = np.hstack((frame_i2_with_fps, frame_p2_numpy, frame_o2_with_fps))

            video.write(float2int(stacked))

            frame_i1 = frame_i2_numpy
            frame_o1 = frame_o2

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release() 
    cv2.destroyAllWindows()
    video.release()

    