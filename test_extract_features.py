import time
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import cooler
import numpy as np
import copy
import os,sys

import model
from utils import operations
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

# 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'
# data from ftp://cooler.csail.mit.edu/coolers/hg19/
def addAtPos(mat1, mat2, xypos = (0,0)):
    pos_v, pos_h = xypos[0], xypos[1]  # offset
    v1 = slice(max(0, pos_v), max(min(pos_v + mat2.shape[0], mat1.shape[0]), 0))
    h1 = slice(max(0, pos_h), max(min(pos_h + mat2.shape[1], mat1.shape[1]), 0))
    v2 = slice(max(0, -pos_v), min(-pos_v + mat1.shape[0], mat2.shape[0]))
    h2 = slice(max(0, -pos_h), min(-pos_h + mat1.shape[1], mat2.shape[1]))
    mat1[v1, h1] += mat2[v2, h2]
    return mat1

def extract_features(path='./data',
            raw_path='raw',
            raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
            model_path=None,
            sr_path='output',
            chromosome='22',
            scale=4,
            len_size=200,
            genomic_distance=2000000,
            start=None, end=None):
    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('-')[1] + '_' + raw_file.split('-')[2] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'extract_features')
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)

    # get generator model
    if model_path is None:
        gan_model_weights_path = './our_model/saved_model/gen_model_' + \
            str(len_size)+'/gen_weights'
    else:
        gan_model_weights_path = model_path
    Generator = model.make_generator_model(len_high_size=len_size, scale=4)
    Generator.load_weights(gan_model_weights_path)
    print(Generator)

    name = os.path.join(path, raw_path, raw_file)
    c = cooler.Cooler(name)
    resolution = c.binsize
    mat = c.matrix(balance=True).fetch('chr'+chromosome)

    [Mh, idx] = operations.remove_zeros(mat)
    print('Shape HR: {}'.format(Mh.shape), end='\t')

    if start is None:
        start = 0
    if end is None:
        end = Mh.shape[0]

    Mh = Mh[start:end, start:end]
    print('MH: {}'.format(Mh.shape), end='\t')

    Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
    print('ML: {}'.format(Ml.shape))

    # Normalization
    # the input should not be type of np.matrix!
    Ml = np.asarray(Ml)
    Mh = np.asarray(Mh)
    Ml, Dl = operations.scn_normalization(Ml, max_iter=3000)
    print('Dl shape:{}'.format(Dl.shape))
    Mh, Dh = operations.scn_normalization(Mh, max_iter=3000)
    print('Dh shape:{}'.format(Dh.shape))

    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))
    # residual = Mh.shape[0] % int(len_size/2)
    # print('residual: {}'.format(residual))

    hic_hr, index_1d_2d, index_2d_1d = operations.divide_pieces_hic( Mh, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr = np.asarray(hic_hr, dtype=np.float32)
    print('shape hic_hr front: ', hic_hr.shape)
    true_hic_hr = hic_hr
    print('shape true hic_hr: ', true_hic_hr.shape)

    hic_lr, _, _ = operations.divide_pieces_hic( Ml, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr = np.asarray(hic_lr, dtype=np.float32)
    print('shape hic_lr: ', hic_lr.shape)
    hic_lr_ds = tf.data.Dataset.from_tensor_slices( hic_lr[..., np.newaxis]).batch(9)
    predict_hic_hr = None
    for i, input_data in enumerate(hic_lr_ds):
        [_, _, tmp, _, _] = Generator(input_data, training=False)
        if predict_hic_hr is None:
            predict_hic_hr = tmp.numpy()
        else:
            predict_hic_hr = np.concatenate( (predict_hic_hr, tmp.numpy()), axis=0)

    layer_name = 'dsd_x2'
    for i, data in enumerate(hic_lr_ds):
        intermediate_layer_model = keras.Model(inputs=Generator.get_layer(layer_name).input,
                                        outputs=Generator.get_layer(layer_name).output)
        intermediate_x2 = intermediate_layer_model(data)

    layer_name = 'dsd_x4'
    for i, data in enumerate(hic_lr_ds):
        intermediate_layer_model = keras.Model(inputs=Generator.get_layer(layer_name).input,
                                    outputs=Generator.get_layer(layer_name).output)
        intermediate_x4 = intermediate_layer_model(data)

    predict_hic_hr = np.squeeze(predict_hic_hr, axis=3)
    print('Shape of prediction front: ', predict_hic_hr.shape)

    file_path = os.path.join(directory_sr, sr_file+'_chr'+chromosome)
    np.savez_compressed(file_path+'.npz', predict_hic=predict_hic_hr, true_hic=true_hic_hr,
                        index_1D_2D=index_1d_2d, index_2D_1D=index_2d_1d,
                        start_id=start, end_id=end, residual=0)

    predict_hic_hr_merge = operations.merge_hic(predict_hic_hr, index_1D_2D=index_1d_2d, max_distance=max_boundary)
    print('Shape of merge predict hic HR', predict_hic_hr_merge.shape)

    true_hic_hr_merge = operations.merge_hic( true_hic_hr, index_1D_2D=index_1d_2d, max_distance=max_boundary)
    print('Shape of merge true hic HR: {}'.format(true_hic_hr_merge.shape))

    # recover M from scn to origin
    Mh = operations.scn_recover(Mh, Dh)
    true_hic_hr_merge = operations.scn_recover(true_hic_hr_merge, Dh)
    predict_hic_hr_merge = operations.scn_recover(predict_hic_hr_merge, Dh)

    # remove diag and off diag
    k = max_boundary.astype(int)
    Mh = operations.filter_diag_boundary(Mh, diag_k=0, boundary_k=k)
    true_hic_hr_merge = operations.filter_diag_boundary(true_hic_hr_merge, diag_k=0, boundary_k=k)
    predict_hic_hr_merge = operations.filter_diag_boundary(predict_hic_hr_merge, diag_k=0, boundary_k=k)

    print('sum Mh:', np.sum(np.abs(Mh)))
    print('sum true merge:', np.sum(np.abs(true_hic_hr_merge)))
    print('sum pred merge:', np.sum(np.abs(predict_hic_hr_merge)))
    diff = np.abs(Mh-predict_hic_hr_merge)
    print('sum Mh - pred square error: {:.5}'.format(np.sum(diff**2)))
    diff = np.abs(true_hic_hr_merge-predict_hic_hr_merge)
    print('sum true merge - pred square error: {:.5}'.format(np.sum(diff**2)))
    diff = np.abs(Mh-true_hic_hr_merge)
    print('sum Mh - true merge square error: {:.5}'.format(np.sum(diff**2)))

    compact = idx
    file = 'predict_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file), hic=predict_hic_hr_merge, compact=compact)
    print('Saving file: {}, at {}'.format(file, directory_sr))
    file = 'true_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file), hic=Mh, compact=compact)
    print('Saving file: {}, at {}'.format(file, directory_sr))


    # predict_hic_hr_merge = predict_hic_hr_merge[::10, ::10]
    # Mh = Mh[::10, ::10]
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    # , cmap='RdBu_r'
    ax = axs[0].imshow(np.log1p(predict_hic_hr_merge), cmap='OrRd')
    axs[0].set_title('predict')
    ax = axs[1].imshow(np.log1p(Mh), cmap='OrRd')  # , cmap='RdBu_r'
    axs[1].set_title('true')
    plt.tight_layout()
    fig.colorbar(ax, ax=axs, shrink=0.3)
    """cmap = mpl.cm.OrRd
    norm = mpl.colors.Normalize(vmin=0, vmax=0.7)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.3)"""
    output = os.path.join(directory_sr, 'prediction_chr{}_{}_{}.png'.format(chromosome, start, end))
    plt.savefig(output, format='png')

    nr,nc = 6,8
    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(25, 20))
    interm = intermediate_x2.numpy()
    interm = np.squeeze(interm, axis=0)
    interm = (interm-interm.min())/(interm.max()-interm.min())
    sum_interm = np.sum(interm, axis=(0,1))
    interm = interm[:,:, sum_interm.argsort()]
    interm = interm[:,:,::-1]
    print(interm.shape)
    for i in np.arange(0, nr):
        for j in np.arange(0, nc):
            idx = 40 + (i*nc+j)*2
            if idx > interm.shape[2]:
                continue
            m = interm[:,:, idx]
            m = np.squeeze(m)
            pcm = axs[i, j].imshow(np.log1p(m), cmap='OrRd')
    plt.tight_layout()
    fig.colorbar(pcm, ax=axs, shrink=0.3)
    output = os.path.join(directory_sr, 'features_x2_chr{}_{}_{}.png'.format(chromosome, start, end))
    plt.savefig(output, format='png')

    nr,nc = 5,7
    fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(25, 20))
    interm = intermediate_x4.numpy()
    interm = np.squeeze(interm, axis=0)
    interm = (interm-interm.min())/(interm.max()-interm.min())
    sum_interm = np.sum(interm, axis=(0,1))
    interm = interm[:,:, sum_interm.argsort()]
    interm = interm[:,:,::-1]
    for i in np.arange(0, nr):
        for j in np.arange(0, nc):
            idx = 10 + (i*nc+j)
            if idx > interm.shape[2]:
                continue
            m = interm[:,:, idx]
            m = np.squeeze(m)
            pcm = axs[i, j].imshow(np.log1p(m), cmap='OrRd')

    plt.tight_layout()
    """cmap = mpl.cm.seismic
    norm = mpl.colors.Normalize(vmin=0, vmax=0.7)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.3)"""
    fig.colorbar(pcm, ax=axs, shrink=0.3)
    output = os.path.join(directory_sr, 'features_x4_chr{}_{}_{}.png'.format(chromosome, start, end))
    plt.savefig(output, format='png')


if __name__ == '__main__':
    root = operations.redircwd_back_projroot(project_name='refine_resolution')
    data_path = os.path.join(root, 'data')
    max_dis = 2000000
    len_size = 400
    chrom = str(sys.argv[1])
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    # raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    extract_features(path=data_path,
            raw_path='raw',
            raw_file=raw_hic,
            chromosome=chrom,
            scale=4,
            len_size=len_size,
            sr_path='_'.join(['output', 'ours', str(max_dis), str(len_size)]),
            genomic_distance=2000000, start=start, end=end)
