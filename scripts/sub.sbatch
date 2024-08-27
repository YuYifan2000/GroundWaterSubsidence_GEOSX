#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=job_combined.log
bash /home/groups/beroza/yuyifan/CPU/GEOS/host-configs/Stanford/sherlock-modules.sh
export HDF5_DISABLE_VERSION_CHECK=1

srun /home/groups/beroza/yuyifan/CPU/GEOS/build-sherlock-gcc10-ompi4.1.2-openblas0.3.10-release/bin/geosx -i pumping.xml

# srun /home/groups/beroza/yuyifan/CPU/GEOS/build-sherlock-gcc10-ompi4.1.2-openblas0.3.10-release/bin/geosx -i impoundment.xml
# srun python3 impoundment_figure.py
# srun /home/groups/beroza/yuyifan/CPU/GEOS/build-sherlock-gcc10-ompi4.1.2-openblas0.3.10-release/bin/geosx -i terzaghi.xml
# srun python3 poromechanicsFigure.py
# srun /home/groups/beroza/yuyifan/CPU/GEOS/build-sherlock-gcc10-ompi4.1.2-openblas0.3.10-release/bin/geosx -i 3D_ter.xml

