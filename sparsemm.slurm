#!/bin/bash
#SBATCH --job-name="mmgw12-csr--sum-mm"
#SBATCH -o hotspots
#SBATCH -e error.err
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par6.q

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/vtune_amplifier_2018.2.0.551022/bin64/amplxe-cl -collect hotspots ./sparsemm c2_sum.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect roofline -- ./sparsemm c2_sum_.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix