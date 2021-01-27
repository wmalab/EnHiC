"""
call 3DChromatin_ReplicateQC to qualify Hi-C matrix


3DChromatin_ReplicateQC run_all 
--metadata_samples examples/metadata.samples 
--metadata_pairs examples/metadata.pairs 
--bins examples/Bins.w50000.bed.gz 
--outdir examples/output 
*   This package need to switch env to 
    3dchromatin_replicate_qc
"""


from .operations import *
import numpy as np
import sys
import io
import os
import subprocess


def run_hicrep(script,
               f1,
               f2,
               bedfile,
               output_path='./',
               maxdist=int(2000000),
               resolution=int(10000),
               h=int(20),
               m1name='m1',
               m2name='m2'):

    cmd = ["Rscript", "--vanilla", script, f1, f2, output_path, str(maxdist), str(resolution), bedfile, str(h), m1name, m2name]
    print(' '.join(cmd))
    process = subprocess.run(cmd, check=True)

    # os.system(' '.join(cmd))
    # proc = subprocess.call([str(' '.join(cmd))],stdout=subprocess.PIPE)
    # stdout_value = proc.wait()

def fit_shape(m1, m2):
    if(m1.shape[0]!=m2.shape[0]):
        mlen = min(m1.shape[0], m2.shape[0])
        m1 = m1[0:mlen, 0:mlen]
        m2 = m2[0:mlen, 0:mlen]
    return m1, m2

def run_mae(mat1, mat2):
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    m1, m2 = fit_shape(m1, m2)
    mae  = np.abs(m1 - m2).mean(axis=None)
    print('m1 sum: {}, m2 sum: {},mae: {}'.format(m1.sum(), m2.sum(), mae))
    return mae

def run_mse(mat1, mat2):
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    m1, m2 = fit_shape(m1, m2)
    mse = ((m1 - m2)**2).mean(axis=None)
    print('m1 sum: {}, m2 sum: {}, mse: {}'.format(m1.sum(), m2.sum(), mse))
    return mse