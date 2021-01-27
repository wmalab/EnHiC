import sys
import os
import qualify
from utils import operations
chromosome = str(sys.argv[1])

root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
# raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
raw_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
len_size = 200
max_dis = 2000000

input_path,sr_file = qualify.configure_our_model(path=os.path.join(root_dir, 'data'),
                               raw_file=raw_file,
                               sr_path = '_'.join(['output','ours',str(max_dis), str(len_size)]),
                               chromosome=chromosome,
                               genomic_distance=2000000,
                               resolution=10000)
file1 = os.path.join(input_path, sr_file+'_contact_true.gz')
file2 = os.path.join(input_path, sr_file+'_contact_predict.gz')
m1name = 'true_{}'.format(chromosome)
m2name = 'predict_{}'.format(chromosome)
output_path = os.path.join(input_path, sr_file+'_scores')
bedfile = os.path.join(input_path, sr_file+'.bed.gz')
script = os.path.join(root_dir, 'our_model', 'utils','hicrep_wrapper.R')
h_list = [20]#, 40, 60, 80]
for h in h_list:
    print('h: ', h)
    output = output_path+ str(h)+'.txt'
    qualify.score_hicrep(file1=file1, file2=file2,
                     bedfile=bedfile, output_path=output, script=script, h=h, m1name=m1name, m2name=m2name)