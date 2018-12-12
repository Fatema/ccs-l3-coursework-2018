#!/bin/bash
#SBATCH --job-name="mmgw12-csr-sum-mm"
#SBATCH -o mm_sum
#SBATCH -e mm_sum.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par6.q

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect roofline -- ./sparsemm c3_sum_.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix