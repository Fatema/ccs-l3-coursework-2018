#!/bin/bash
#SBATCH --job-name="mmgw12-csr-mm-hotspots"
#SBATCH -o hotspots
#SBATCH -e error.err
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p test.q
#SBATCH -t 00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
source /etc/profile.d/modules.sh

module load pgi/17.9
module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/vtune_amplifier_2018.2.0.551022/bin64/amplxe-cl -collect hotspots ./sparsemm c2.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect roofline -- ./sparsemm c2_.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/large-CG1-mass-3D.matrix