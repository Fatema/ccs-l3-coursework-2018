#!/bin/bash
#SBATCH --job-name="mmgw12-csr-mm-hotspots"
#SBATCH -o hotspots
#SBATCH -e error.err
#SBATCH -p par6.q
#SBATCH -t 00:05:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL

module load pgi/17.9
module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/vtune_amplifier_2018.2.0.551022/bin64/amplxe-cl -collect hotspots ./sparsemm c2.matrix /ddn/home/mmgw12/matrices-lg/DG4-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG4-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG4-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG4-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG4-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG4-mass-3D.matrix 