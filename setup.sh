#! /bin/bash
module load pgi/17.9
module load slurm
module load intel/xe_2018.2
source /ddn/apps/Cluster-Apps/intel/xe_2018.2/vtune_amplifier/amplxe-vars.sh
source /ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor/advixe-vars.sh

export MLG=~/matrices-lg
export MSM=~/matrices-sm
export MVLG=~/matrices-vlg