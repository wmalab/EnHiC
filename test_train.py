import glob
import imageio
import time
import datetime
from IPython import display
import matplotlib.pyplot as plt
import cooler
import numpy as np
import copy
import os
import sys
import shutil
import logging
from EnHiC import fit
from EnHiC.utils.operations import sampling_hic
from EnHiC.utils.operations import divide_pieces_hic, merge_hic
from EnHiC.utils.operations import redircwd_back_projroot
import tensorflow as tf


def gethic_data(chr_list, data_path, input_path, input_file):
    hr_file_list = []
    for chri in chr_list:
        path = os.path.join(data_path, input_path,
                            input_file, 'HR', 'chr'+chri)
        if not os.path.exists(path):
            continue
        for file in os.listdir(path):
            if file.endswith(".npz"):
                pathfile = os.path.join(path, file)
                hr_file_list.append(pathfile)
    hr_file_list.sort()
    logging.info("chromosomes list: {}".format(chr_list))
    hic_hr = None
    hic_lr = None
    for hr_file in hr_file_list:
        lr_file = hr_file.replace('HR', 'LR')
        logging.info(hr_file)
        logging.info(lr_file)
        if (not os.path.exists(hr_file)) or (not os.path.exists(lr_file)):
            continue
        with np.load(hr_file, allow_pickle=True) as data:
            if hic_hr is None:
                hic_hr = data['hic']
            else:
                hic_hr = np.concatenate((hic_hr, data['hic']), axis=0)
        with np.load(lr_file, allow_pickle=True) as data:
            if hic_lr is None:
                hic_lr = data['hic']
            else:
                hic_lr = np.concatenate((hic_lr, data['hic']), axis=0)
    return hic_hr, hic_lr


if __name__ == '__main__':
    # the size of input
    len_size = int(sys.argv[1])  # 40, 128, 200
    scale = 4
    # genomic_disstance is used for input path, nothing to do with model
    genomic_distance = int(sys.argv[2])  # 2000000, 2560000
    EPOCHS = 1
    BATCH_SIZE = 4
    root_path = redircwd_back_projroot(project_name='refine_resolution')
    data_path = os.path.join(root_path, 'data')
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    # raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    input_path = '_'.join(['input', 'EnHiC', str(genomic_distance), str(len_size)])
    input_file = raw_hic.split('-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('-')[2] + '_' + raw_hic.split('.')[1]

    log_dir = os.path.join(root_path, 'logs', 'model')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)

    # train ['1', '2', '3', '4', '5','6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    # valid ['17', '18']
    # test  ['19', '20', '21', '22', 'X']
    train_chr_list = ['22'] # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    valid_chr_list = ['22'] # ['17', '18']

    hic_hr, hic_lr = gethic_data(
        train_chr_list, data_path, input_path, input_file)
    hic_lr = np.asarray(hic_lr).astype(np.float32)
    hic_hr = np.asarray(hic_hr).astype(np.float32)
    logging.info("train hic_lr shape: {}".format(hic_lr.shape))
    logging.info("train hic_hr shape: {}".format(hic_hr.shape))
    train_data = tf.data.Dataset.from_tensor_slices(
        (hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)

    hic_hr, hic_lr = gethic_data(valid_chr_list, data_path, input_path, input_file)
    hic_lr = np.asarray(hic_lr).astype(np.float32)
    hic_hr = np.asarray(hic_hr).astype(np.float32)
    logging.info("valid hic_lr shape: {}".format(hic_lr.shape))
    logging.info("valid hic_hr shape: {}".format(hic_hr.shape))
    valid_data = tf.data.Dataset.from_tensor_slices(
        (hic_lr[..., np.newaxis], hic_hr[..., np.newaxis])).batch(BATCH_SIZE)

    #load_model_dir = os.path.join(root_path, 'EnHiC', 'saved_model')
    saved_model_dir = os.path.join(root_path, 'saved_model')
    fit.train(train_data=train_data, valid_data=valid_data, len_size=len_size, scale=scale,
        EPOCHS=EPOCHS, root_path=root_path,
        load_model_dir=None, saved_model_dir=saved_model_dir, log_dir=None,
        summary=True)
