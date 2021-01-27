from EnHiC import prepare_data
import sys
from EnHiC.utils import operations
import os

# raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
data_fp = os.path.join(operations.redircwd_back_projroot(project_name='refine_resolution'), 'data')
config = prepare_data.configure(len_size=int(sys.argv[2]), genomic_distance=int(sys.argv[3]), dataset_path=data_fp, raw_hic=raw_hic)
chromosome_list = [str(sys.argv[1])]

for chri in chromosome_list:
    prepare_data.save_samples(config, chromosome=chri)