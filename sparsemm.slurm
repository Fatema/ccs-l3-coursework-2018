#!/bin/bash
#SBATCH --job-name="mmgw12-csr-mm"
#SBATCH -o mm
#SBATCH -e mm.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par6.q

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect roofline -project-dir /ddn/home/mmgw12/ccs-l3-coursework-2018 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm --binary c_vlg_.matrix /ddn/home/mmgw12/matrices-vlg/CG1-laplace-3D-bigger.matrix /ddn/home/mmgw12/matrices-vlg/CG1-mass-3D-bigger.matrix