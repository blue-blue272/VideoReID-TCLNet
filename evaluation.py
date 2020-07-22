from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np
sys.path.append('tools/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from eval_metrics import evaluate


def extract(model, args, vids, use_gpu):
    n, c, f, h, w = vids.size()
    assert(n == 1)

    feat = []
    for i in range(int(math.ceil(f*1.0/args.test_frames))):
        clip = vids[:, :, i*args.test_frames:(i+1)*args.test_frames, :, :]
        clip = clip.cuda()

        if clip.size()[2] < args.test_frames:
            total_clip = clip
            while total_clip.size()[2] < args.test_frames:
                for idx in range(clip.size()[2]):
                    if total_clip.size()[2] >= args.test_frames:
                        break
                    total_clip = torch.cat((total_clip, clip[:,:,idx:idx+1]), 2)
            clip = total_clip

        assert clip.size(2) == args.test_frames
        output = model(clip) 
        feat.append(output)

    feat = torch.stack(feat, 1)
    feat = feat.mean(1) 

    feat_list = torch.split(feat, 2048, dim=1)
    norm_feat_list = []
    for i, f in enumerate(feat_list):
        f = model.module.bn[i](f) 
        f = F.normalize(f, p=2, dim=1, eps=1e-12)
        norm_feat_list.append(f)
    feat = torch.cat(norm_feat_list, 1)

    return feat

def evaluation(model, args, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if (batch_idx + 1) % 1000==0:
            print("{}/{}".format(batch_idx+1, len(queryloader)))

        qf.append(extract(model, args, vids, use_gpu).squeeze())
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if (batch_idx + 1) % 1000==0:
            print("{}/{}".format(batch_idx+1, len(galleryloader)))

        gf.append(extract(model, args, vids, use_gpu).squeeze())
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    if 'mars' in args.dataset:
        print('process the dataset mars!')
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))

    distmat = - torch.mm(qf, gf.t())
    distmat = distmat.data.cpu()
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    elapsed = round(time.time() - since)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}.".format(elapsed))

    return cmc[0]
