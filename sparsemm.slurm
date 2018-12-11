#!/bin/bash
#SBATCH --job-name="mmgw12-csr-mm"
#SBATCH -o c.matrix
#SBATCH -e error.err
#SBATCH -p par6.q
#SBATCH -t 00:05:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
source /etc/profile.d/modules.sh
module load intel/xe_2018.2
source /ddn/apps/Cluster-Apps/intel/xe_2018.2/vtune_amplifier/amplxe-vars.sh
source /ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor/advixe-vars.sh

./a.out