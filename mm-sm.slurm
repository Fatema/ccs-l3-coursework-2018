#!/bin/bash
#SBATCH --job-name="mmgw12-csr-sm-mm"
#SBATCH -o mm-sm
#SBATCH -e mm-sm.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=fatema.alkhanaizi@durham.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -p par6.q

source /etc/profile.d/modules.sh

module load intel/xe_2018.2

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_sm/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_sm/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix

rm /ddn/home/mmgw12/mm_sum_sm/DG1/c_sm_sum_.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_sm/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_sm/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix

rm /ddn/home/mmgw12/mm_sum_sm/DG2/c_sm_sum_.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sum_sm/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sum_sm/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_sum_.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix

rm /ddn/home/mmgw12/mm_sum_sm/DG3/c_sm_sum_.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sm/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sm/DG1 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/DG1-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG1-mass-2D.matrix

rm /ddn/home/mmgw12/mm_sm/DG1/c_sm_.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sm/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sm/DG2 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/DG2-ip-laplace-2D.matrix /ddn/home/mmgw12/matrices-sm/DG2-mass-2D.matrix

rm /ddn/home/mmgw12/mm_sm/DG2/c_sm_.matrix

/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect survey -project-dir /ddn/home/mmgw12/mm_sm/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix
/ddn/apps/Cluster-Apps/intel/xe_2018.2/advisor_2018.2.0.551025/bin64/advixe-cl -collect tripcounts -flop -project-dir /ddn/home/mmgw12/mm_sm/DG3 -- /ddn/home/mmgw12/ccs-l3-coursework-2018/sparsemm c_sm_.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-laplace-3D.matrix /ddn/home/mmgw12/matrices-sm/small-CG1-mass-3D.matrix

rm /ddn/home/mmgw12/mm_sm/DG3/c_sm_.matrix
