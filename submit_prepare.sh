
chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X')
#chr=('22')
#contact=('1' '2' '3' '4' '5' '6' '7' '8')
len_size=${1}
#40, 80, 128
boundary=${2}
#'2000000'
#rm slurm-data-*.out
for c in "${chr[@]}"; do
    echo sbatch bash_prepare_data.sh $c $len_size $boundary
    sbatch bash_prepare_data.sh $c $len_size $boundary
done
