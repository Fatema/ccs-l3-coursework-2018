#!/bin/bash
#SBATCH --job-name="mmgw12-csr-lg-mm"
#SBATCH -o mm-lg
#SBATCH -e mm-lg.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par6.q

export INTEL_LICENSE_FILE=~/NCOM_L___S4ZD-KPHBLZSL.lic

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_lg/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_lg/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_lg/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_lg/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_lg/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_lg/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_sum_.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_lg/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_lg/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG1-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_lg/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_lg/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG2-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG2-mass-3D.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_lg/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_lg/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_lg_.matrix /ddn/home/mmgw12/matrices-lg/DG3-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-lg/DG3-mass-3D.matrix
