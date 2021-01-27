
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import cooler
from EnHiC.utils import operations
from EnHiC import model, fit
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

def run(path='./data',
        raw_path='raw',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        model_path=None,
        sr_path='output',
        chromosome='22',
        scale=4,
        len_size=200,
        genomic_distance=2000000,
        start=None, end=None, draw_out=False):

    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('-')[1] + \
        '_' + raw_file.split('-')[2] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)

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
    residual = Mh.shape[0] % int(len_size/2)
    print('residual: {}'.format(residual))

    hic_hr_front, index_1d_2d_front, index_2d_1d_front = operations.divide_pieces_hic(
        Mh[0:-residual, 0:-residual], block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr_front = np.asarray(hic_hr_front, dtype=np.float32)
    print('shape hic_hr front: ', hic_hr_front.shape)
    true_hic_hr_front = hic_hr_front
    print('shape true hic_hr: ', true_hic_hr_front.shape)

    hic_hr_offset, index_1d_2d_offset, index_2d_1d_offset = operations.divide_pieces_hic(
        Mh[residual:, residual:], block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr_offset = np.asarray(hic_hr_offset, dtype=np.float32)
    print('shape hic_hr offset: ', hic_hr_offset.shape)
    true_hic_hr_offset = hic_hr_offset
    print('shape true hic_hr: ', true_hic_hr_offset.shape)

    Ml_front = Ml[0:-residual, 0:-residual]
    hic_lr_front, _, _ = operations.divide_pieces_hic(
        Ml_front, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr_front = np.asarray(hic_lr_front, dtype=np.float32)
    print('shape hic_lr: ', hic_lr_front.shape)
    hic_lr_ds = tf.data.Dataset.from_tensor_slices(
        hic_lr_front[..., np.newaxis]).batch(9)

    predict_hic_hr_front = fit.predict(model_path, len_size, scale, hic_lr_ds)
    predict_hic_hr_front = np.squeeze(predict_hic_hr_front, axis=3)
    print('Shape of prediction front: ', predict_hic_hr_front.shape)

    file_path = os.path.join(directory_sr, sr_file+'_chr'+chromosome)
    np.savez_compressed(file_path+'_front.npz', predict_hic=predict_hic_hr_front, true_hic=true_hic_hr_front,
                        index_1D_2D=index_1d_2d_front, index_2D_1D=index_2d_1d_front,
                        start_id=start, end_id=end, residual=0)

    predict_hic_hr_merge_front = operations.merge_hic(
        predict_hic_hr_front, index_1D_2D=index_1d_2d_front, max_distance=max_boundary)
    print('Shape of merge predict hic HR front',
          predict_hic_hr_merge_front.shape)

    Ml_offset = Ml[residual:, residual:]
    hic_lr_offset, _, _ = operations.divide_pieces_hic(
        Ml_offset, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr_offset = np.asarray(hic_lr_offset, dtype=np.float32)
    print('Shape hic_lr_offset: ', hic_lr_offset.shape)
    hic_lr_ds = tf.data.Dataset.from_tensor_slices(
        hic_lr_offset[..., np.newaxis]).batch(9)

    predict_hic_hr_offset = fit.predict(model_path, len_size, scale, hic_lr_ds)
    predict_hic_hr_offset = np.squeeze(predict_hic_hr_offset, axis=3)
    print('Shape of prediction offset: ', predict_hic_hr_offset.shape)

    file_path = os.path.join(directory_sr, sr_file+'_chr'+chromosome)
    np.savez_compressed(file_path+'_offset.npz', predict_hic=predict_hic_hr_offset, true_hic=true_hic_hr_offset,
                        index_1D_2D=index_1d_2d_offset, index_2D_1D=index_2d_1d_offset,
                        start_id=start, end_id=end, residual=residual)
    predict_hic_hr_merge_offset = operations.merge_hic(
        predict_hic_hr_offset, index_1D_2D=index_1d_2d_offset, max_distance=max_boundary)
    print('Shape of merge predict hic hr offset: ',
          predict_hic_hr_merge_offset.shape)

    predict_hic_hr_merge = np.zeros(Mh.shape)
    predict_hic_hr_merge = addAtPos(
        predict_hic_hr_merge, predict_hic_hr_merge_front, (0, 0))
    predict_hic_hr_merge = addAtPos(
        predict_hic_hr_merge, predict_hic_hr_merge_offset, (residual, residual))

    ave = np.ones_like(predict_hic_hr_merge)
    twice = np.ones(shape=(Mh.shape[0]-2*residual, Mh.shape[1]-2*residual))
    ave = addAtPos(ave, twice, (residual, residual))
    predict_hic_hr_merge = predict_hic_hr_merge/ave

    true_hic_hr_merge_front = operations.merge_hic(
        true_hic_hr_front, index_1D_2D=index_1d_2d_front, max_distance=max_boundary)
    true_hic_hr_merge_offset = operations.merge_hic(
        true_hic_hr_offset, index_1D_2D=index_1d_2d_offset, max_distance=max_boundary)
    true_hic_hr_merge = np.zeros(Mh.shape)
    true_hic_hr_merge = addAtPos(
        true_hic_hr_merge, true_hic_hr_merge_front, (0, 0))
    true_hic_hr_merge = addAtPos(
        true_hic_hr_merge, true_hic_hr_merge_offset, (residual, residual))
    true_hic_hr_merge = true_hic_hr_merge/ave
    print('Shape of true merge hic hr: {}'.format(true_hic_hr_merge.shape))

    # recover M from scn to origin
    Mh = operations.scn_recover(Mh, Dh)
    true_hic_hr_merge = operations.scn_recover(true_hic_hr_merge, Dh)
    predict_hic_hr_merge = operations.scn_recover(predict_hic_hr_merge, Dh)

    # remove diag and off diag
    k = max_boundary.astype(int)
    Mh = operations.filter_diag_boundary(Mh, diag_k=0, boundary_k=k)
    true_hic_hr_merge = operations.filter_diag_boundary(
        true_hic_hr_merge, diag_k=0, boundary_k=k)
    predict_hic_hr_merge = operations.filter_diag_boundary(
        predict_hic_hr_merge, diag_k=0, boundary_k=k)

    print('sum Mh:', np.sum(np.abs(Mh)))
    print('sum true merge:', np.sum(np.abs(true_hic_hr_merge)))
    print('sum pred merge:', np.sum(np.abs(predict_hic_hr_merge)))
    diff = np.abs(Mh-predict_hic_hr_merge)
    print('sum Mh - pred square error: {:.5}'.format(np.sum(diff**2)))
    diff = np.abs(true_hic_hr_merge-predict_hic_hr_merge)
    print('sum true merge - pred square error: {:.5}'.format(np.sum(diff**2)))
    diff = np.abs(Mh-true_hic_hr_merge)
    print('sum Mh - true merge square error: {:.5}'.format(np.sum(diff**2)))

    directory_sr = os.path.join(path, sr_path, sr_file, 'SR')
    compact = idx[0:-residual]
    file = 'predict_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file),
                        hic=predict_hic_hr_merge, compact=compact)
    print('Saving file: {}, at {}'.format(file, directory_sr))
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)
    file = 'true_chr{}_{}.npz'.format(chromosome, resolution)
    np.savez_compressed(os.path.join(directory_sr, file),
                        hic=Mh, compact=compact)
    print('Saving file: {}, at {}'.format(file, directory_sr))


    if draw_out:
        predict_hic_hr_merge = predict_hic_hr_merge[::10, ::10]
        Mh = Mh[::10, ::10]
        fig, axs = plt.subplots(1, 2, figsize=(8, 15))
        # , cmap='RdBu_r'
        ax = axs[0].imshow(np.log1p(predict_hic_hr_merge), cmap='RdBu')
        axs[0].set_title('predict')
        ax = axs[1].imshow(np.log1p(Mh), cmap='RdBu')  # , cmap='RdBu_r'
        axs[1].set_title('true')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    chromosome = str(sys.argv[1]) # '22'
    len_size = int(sys.argv[2])  # 200
    max_dis = 2000000

    root_dir = operations.redircwd_back_projroot( project_name='EnHiC')
    raw_hic = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    # raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'

    run(path=os.path.join(root_dir, 'data'),
        raw_path='raw',
        raw_file=raw_hic,
        model_path=os.path.join(root_dir,'saved_model', 'gen_model_{}'.format(len_size), 'gen_weights'),
        chromosome=chromosome,
        scale=4,
        len_size=len_size,
        sr_path='_'.join(['output', 'EnHiC', str(max_dis), str(len_size)]),
        genomic_distance=2000000,
        start=None, end=None, draw_out=True)
