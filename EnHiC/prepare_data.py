import numpy as np
import os
import sys
import cooler
import wget

from utils import operations

# https://github.com/mirnylab/cooler-binder/blob/master/cooler_api.ipynb
# data from ftp://cooler.csail.mit.edu/coolers/hg19/


def configure(len_size=None, genomic_distance=None, methods_name='EnHiC',
              dataset_path=None,
              raw_path='raw',
              raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
              input_path='input',
              output_path='output'):

    resolution = None  # assigned by cooler binsizes
    scale = 4
    if len_size is None:
        len_size = 40
    block_size = 2048  # number of entries in one file
    if genomic_distance is None:
        genomic_distance = 200000
    if dataset_path is None:
        # assume current directory is the root of project
        # pathto/proj/data
        # pathto/proj/our_method
        dataset_path = os.path.join(operations.redircwd_back_projroot(
            project_name='refine_resolution'), 'data')

    print('data path: ', dataset_path)
    input_file = raw_hic.split( '-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('-')[2] + '_' + raw_hic.split('.')[1]
    input_path = '_'.join(
        [input_path, methods_name, str(genomic_distance), str(len_size)])
    output_file = input_file
    output_path = '_'.join(
        [output_path, methods_name, str(genomic_distance), str(len_size)])

    # load raw hic matrix
    file = os.path.join(dataset_path, raw_path, raw_hic)
    print('raw hic data: ', file)
    if not os.path.exists(file):
        os.makedirs(os.path.join(dataset_path, raw_path), exist_ok=True)
        url = 'ftp://cooler.csail.mit.edu/coolers/hg19/'+raw_hic
        print(url)
        file = wget.download(url, file)
    cool_hic = cooler.Cooler(file)
    resolution = cool_hic.binsize
    return cool_hic, resolution, scale, len_size, genomic_distance,\
        block_size, dataset_path, \
        [raw_path, raw_hic], \
        [input_path, input_file], \
        [output_path, output_file]


def save_samples(configure=None, chromosome=None):
    cool_hic, resolution, scale, len_size, genomic_distance, \
        block_size, data_path, \
        [raw_path, raw_hic], \
        [input_path, input_file], \
        [output_path, output_file] = configure
    chromosome = 'chr' + chromosome
    mat = cool_hic.matrix(balance=True).fetch(chromosome)
    Mh, _ = operations.remove_zeros(mat)
    #Mh = Mh[0:512, 0:512]
    print('MH: ', Mh.shape)
    Ml = operations.sampling_hic(Mh, scale**2, fix_seed=True)
    print('ML: ', Ml.shape)

    # Normalization
    # the input should not be type of np.matrix!
    Ml = np.asarray(Ml)
    Mh = np.asarray(Mh)
    norm_Ml,Dl = operations.scn_normalization(Ml, max_iter=1000)
    norm_Mh,Dh = operations.scn_normalization(Mh, max_iter=1000)
    operations.check_scn(Ml, norm_Ml,Dl)
    operations.check_scn(Mh, norm_Mh,Dh)
    Ml = norm_Ml
    Mh = norm_Mh

    # min-max norm
    # Ml = np.divide((Ml-Ml.min()), (Ml.max()-Ml.min()), dtype=float, out=np.zeros_like(Ml), where=(Ml.max()-Ml.min()) != 0)
    # Mh = np.divide((Mh-Mh.min()), (Mh.max()-Mh.min()), dtype=float, out=np.zeros_like(Mh), where=(Mh.max()-Mh.min()) != 0)

    max_boundary = None
    if genomic_distance is not None:
        max_boundary = np.ceil(genomic_distance/(resolution))

    hic_hr, _, _ = operations.divide_pieces_hic(
        Mh, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_hr = np.asarray(hic_hr)
    print('shape hic_hr: ', hic_hr.shape)
    hic_lr, _, _ = operations.divide_pieces_hic(
        Ml, block_size=len_size, max_distance=max_boundary, save_file=False)
    hic_lr = np.asarray(hic_lr)
    print('shape hic_lr: ', hic_lr.shape)

    directory_hr = os.path.join(
        data_path, input_path, input_file, 'HR', chromosome)
    directory_lr = os.path.join(
        data_path, input_path, input_file, 'LR', chromosome)
    directory_sr = os.path.join(
        data_path, output_path, output_file, 'SR', chromosome)
    if not os.path.exists(directory_hr):
        os.makedirs(directory_hr)
    if not os.path.exists(directory_lr):
        os.makedirs(directory_lr)
    if not os.path.exists(directory_sr):
        os.makedirs(directory_sr)

    # random
    len_1d = hic_hr.shape[0]
    idx = np.random.choice(np.arange(len_1d), size=len_1d, replace=False)
    for ibs in np.arange(0, hic_hr.shape[0], block_size):
        start = ibs
        end = min(start+block_size, hic_hr.shape[0])

        idx_sub = idx[start:end]
        hic_m = hic_hr[idx_sub, :, :]
        pathfile = input_file+'_HR_' + chromosome + \
            '_' + str(start) + '-' + str(end-1)
        pathfile = os.path.join(directory_hr, pathfile)
        print(pathfile)
        np.savez_compressed(pathfile+'.npz', hic=hic_m,
                            index_1D_2D=None, index_2D_1D=None)

        hic_m = hic_lr[idx_sub, :, :]
        pathfile = input_file+'_LR_' + chromosome + \
            '_' + str(start) + '-' + str(end-1)
        pathfile = os.path.join(directory_lr, pathfile)
        print(pathfile)
        np.savez_compressed(pathfile+'.npz', hic=hic_m,
                            index_1D_2D=None, index_2D_1D=None)


"""
configure data:
dataset_path-raw
            -input
            -output
"""

if __name__ == '__main__':
    root = operations.redircwd_back_projroot(project_name='refine_resolution')
    raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    # raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    config = configure(raw_hic=raw_hic, len_size=int(sys.argv[2]), methods_name='EnHiC', genomic_distance=int(sys.argv[3]))
    chromosome_list = [str(sys.argv[1])]
    for chri in chromosome_list:
        save_samples(config, chromosome=chri)
