#!/bin/bash
#SBATCH --job-name="mmgw12-csr-sum-mm"
#SBATCH -o mm_sum
#SBATCH -e mm_sum.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par7.q

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm --binary c_vlg_sum_.matrix /ddn/home/mmgw12/matrices-vlg/CG1-laplace-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-laplace-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-laplace-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-mass-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-mass-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-mass-3D-bigger.matrix
