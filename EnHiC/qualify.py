import time
import datetime
import cooler
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cooler

from utils.operations import redircwd_back_projroot
from utils.operations import remove_zeros, merge_hic, filter_diag_boundary, format_bin, format_contact
from utils.operations import scn_normalization, scn_recover, sampling_hic
from utils.quality_hic import run_hicrep, run_mae, run_mse
# 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'


def configure_our_model(
        path='./data',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        sr_path = 'output_ours_2000000_200',
        chromosome='22',
        genomic_distance=2000000,
        resolution=10000):

    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('-')[1] + + '_' + raw_file.split('-')[2] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)

    input_path = directory_sr
    input_file = sr_file+'_chr'+chromosome+'.npz'
    if not os.path.exists(os.path.join(input_path, input_file)):
        print('not input file')

    print('input path: ', input_path, input_file)
    with np.load(os.path.join(input_path, input_file), allow_pickle=True) as data:
        predict_hic = data['predict_hic']
        true_hic = data['true_hic']
        idx_1d_2d = data['index_1D_2D'][()]  # get dict()
        idx_2d_1d = data['index_2D_1D'][()]  # get dict()
        start = data['start_id']
        end = data['end_id']

    print(end+1)
    print(idx_1d_2d)

    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))

    predict_hic_hr_merge = merge_hic(
        predict_hic, index_1D_2D=idx_1d_2d, max_distance=max_boundary)
    print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

    true_hic_hr_merge = merge_hic(
        true_hic, index_1D_2D=idx_1d_2d, max_distance=max_boundary)
    print('shape of merge predict hic hr', predict_hic_hr_merge.shape)


    k = np.ceil(genomic_distance/resolution).astype(int)
    true_hic_hr_merge = filter_diag_boundary(
        true_hic_hr_merge, diag_k=2, boundary_k=k)

    predict_hic_hr_merge = filter_diag_boundary(
        predict_hic_hr_merge, diag_k=2, boundary_k=k)

    true_hic_hr_merge, Dh = scn_normalization(true_hic_hr_merge)
    true_hic_hr_merge = scn_recover(true_hic_hr_merge, Dh)
    predict_hic_hr_merge = scn_recover(predict_hic_hr_merge, Dh)

    print('sum true:', np.sum(np.abs(true_hic_hr_merge)))
    print('sum predict:', np.sum(np.abs(predict_hic_hr_merge)))
    diff = np.abs(true_hic_hr_merge-predict_hic_hr_merge)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))

    '''format_bin(true_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'.bed.gz')
    format_contact(true_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_true.gz')
    format_contact(predict_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_predict.gz')
    '''
    format_bin(true_hic_hr_merge, resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'.bed.gz')
    format_contact(true_hic_hr_merge, resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_true.gz')
    format_contact(predict_hic_hr_merge, resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_predict.gz')

    return input_path, sr_file

def configure_model(
        model='deephic',
        path='./data',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        sr_path = 'output',
        chromosome='22',
        genomic_distance=2000000,
        resolution=10000,
        true_path=None):

    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('-')[1] + '_' + raw_file.split('-')[2] + '_' + raw_file.split('.')[1]
    input_path = os.path.join(path, sr_path, sr_file, 'SR')
    input_file = 'predict_chr'+chromosome+'_{}.npz'.format(resolution)

    if not os.path.exists(os.path.join(input_path, input_file)):
        print('not input file')

    print('input path: ', input_path, input_file)
    with np.load(os.path.join(input_path, input_file), allow_pickle=True) as data:
        predict_hic = data['hic']

    '''if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))'''

    raw_path = os.path.join(path, 'raw')
    file = os.path.join(raw_path, raw_file)
    print('raw hic data: ', file)
    cool_hic = cooler.Cooler(file)
    # resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    true_hic, _ = remove_zeros(mat)
    residue = true_hic.shape[0]%100
    true_hic = true_hic[0:-residue,0:-residue]


    if model == 'hicgan':
        # true_hic = np.log1p(true_hic)
        predict_hic = np.expm1(predict_hic)
    elif model == 'deephic':
        minv = true_hic.min()
        maxv = true_hic.max()
        # true_hic = np.divide((true_hic-minv), (maxv-minv), dtype=float,out=np.zeros_like(true_hic), where=(maxv-minv) != 0)
        predict_hic = predict_hic*(maxv-minv)+minv
    elif model == 'hicsr':
        log_mat = np.log2(true_hic+1)
        # ture_hic = 2*(log_mat/np.max(log_mat)) - 1
        maxv = np.max(log_mat)
        log_predict_hic = (predict_hic+1)/2*maxv
        predict_hic = np.expm1(log_predict_hic)

    k = np.ceil(genomic_distance/resolution).astype(int)
    true_hic = filter_diag_boundary(true_hic, diag_k=2, boundary_k=k)
    predict_hic = filter_diag_boundary(predict_hic, diag_k=2, boundary_k=k)
    predict_hic = predict_hic[np.arange(true_hic.shape[0]), :]
    predict_hic = predict_hic[:, np.arange(true_hic.shape[1])]

    print('shape of predict hic', predict_hic.shape)
    print('shape of true hic', true_hic.shape)

    print('sum true:', np.sum(np.abs(true_hic)))
    print('sum predict:', np.sum(np.abs(predict_hic)))
    diff = np.abs(true_hic - predict_hic)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))

    input_path = os.path.join(input_path, 'chr{}'.format(chromosome))
    os.makedirs(input_path, exist_ok=True)
    format_bin(true_hic, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=os.path.join(input_path, sr_file+'.bed.gz'))
    format_contact(true_hic, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=os.path.join(input_path, sr_file+'_contact_true.gz'))
    format_contact(predict_hic, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=os.path.join(input_path,  sr_file+'_contact_predict.gz'))

    return input_path, sr_file

def score_hicrep(file1,
                 file2,
                 bedfile,
                 output_path,
                 script='./utils/hicrep_wrapper.R',
                 maxdist=int(2000000),
                 resolution=int(10000),
                 h=int(20),
                 m1name='m1',
                 m2name='m2'):
    run_hicrep(script=script, f1=file1, f2=file2,
                           bedfile=bedfile, output_path=output_path, maxdist=maxdist,
                           resolution=resolution,
                           h=h,
                           m1name=m1name,
                           m2name=m2name)

def metric_mae(file1, file2, output_path, model,
                m1name='m1',
                m2name='m2', max_boundary=200, diag_k=2):
    data1 = np.load(file1, allow_pickle=True)
    high_mat = data1['hic']
    data2 = np.load(file2, allow_pickle=True)
    mat = data2['hic']
    high_mat = filter_diag_boundary(high_mat, diag_k=diag_k, boundary_k=max_boundary)
    if 'hicgan' in model:
        # true_hic = np.log1p(true_hic)
        mat = np.expm1(mat)
    elif 'deephic' in model:
        minv = high_mat.min()
        maxv = high_mat.max()
        # true_hic = np.divide((true_hic-minv), (maxv-minv), dtype=float,out=np.zeros_like(true_hic), where=(maxv-minv) != 0)
        mat = mat*(maxv-minv)+minv
        mat = (mat + np.transpose(mat))/2
    elif 'hicsr' in model:
        log_mat = np.log2(high_mat+1)
        # ture_hic = 2*(log_mat/np.max(log_mat)) - 1
        maxv = np.max(log_mat)
        log_predict_hic = (mat+1)/2*maxv
        mat = np.expm1(log_predict_hic)
        mat = (mat + np.transpose(mat))/2

    hic1 = high_mat
    hic2 = filter_diag_boundary(mat, diag_k=diag_k, boundary_k=max_boundary)

    mae = run_mae(mat1=hic1, mat2=hic2)

    # header = 'method1 \t method2 \t mae\n'
    line = '{} \t {} \t {}\n'.format(m1name, m2name, mae)
    fout = open(output_path, 'w+')
    # fout.write(header)
    fout.write(line)
    fout.close()

def metric_mse(file1, file2, output_path, model,
                m1name='m1',
                m2name='m2', max_boundary=200, diag_k=2):
    data1 = np.load(file1, allow_pickle=True)
    high_mat = data1['hic']
    data2 = np.load(file2, allow_pickle=True)
    mat = data2['hic']
    high_mat = filter_diag_boundary(high_mat, diag_k=diag_k, boundary_k=max_boundary)
    if 'hicgan' in model:
        # true_hic = np.log1p(true_hic)
        mat = np.expm1(mat)
    elif 'deephic' in model:
        minv = high_mat.min()
        maxv = high_mat.max()
        # true_hic = np.divide((true_hic-minv), (maxv-minv), dtype=float,out=np.zeros_like(true_hic), where=(maxv-minv) != 0)
        print('maxv: {}, minv: {}'.format(maxv, minv))
        mat = mat*(maxv-minv)+minv
        mat = (mat + np.transpose(mat))/2
    elif 'hicsr' in model:
        log_mat = np.log2(high_mat+1)
        # ture_hic = 2*(log_mat/np.max(log_mat)) - 1
        maxv = np.max(log_mat)
        log_predict_hic = (mat+1)/2*maxv
        mat = np.expm1(log_predict_hic)
        mat = (mat + np.transpose(mat))/2

    hic1 = high_mat # filter_diag_boundary(high_mat, diag_k=diag_k, boundary_k=max_boundary)
    hic2 = filter_diag_boundary(mat, diag_k=diag_k, boundary_k=max_boundary)
    mse = run_mse(mat1=hic1, mat2=hic2)

    # header = 'method1 \t method2 \t mse\n'
    line = '{} \t {} \t {}\n'.format(m1name, m2name, mse)
    fout = open(output_path, 'w+')
    # fout.write(header)
    fout.write(line)
    fout.close()

if __name__ == '__main__':
    root = redircwd_back_projroot(project_name='refine_resolution')
    [input_path, hicfile] = configure_our_model(path = os.path.join(root, 'data'), sr_path = 'output_ours_2000000_200')
    file1 = '"' + input_path+'/' + hicfile + '_contact_true.gz' + '"'
    file2 = '"' +input_path+'/' + hicfile + '_contact_predict.gz'+ '"'
    output = '"' +input_path+'/'+ hicfile + '_scores.txt'+ '"'
    bedfile = '"' +input_path+'/'+ hicfile+ '.bed.gz' + '"'
    script = './our_model/utils/hicrep_wrapper.R'
    score_hicrep(script=script, file1=file1, file2=file2,
                 bedfile=bedfile, output_path=output)
