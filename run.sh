#!/bin/bash
#SBATCH --account=rrg-mtaylor3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000M
#SBATCH --time=80:10:00

#SBATCH --array=0-4   



#SBATCH --job-name=fetch_push
#SBATCH --output=test-%J.out
#SBATCH --array=1-3

cd /home/cmuslima/projects/def-mtaylor3/cmuslima/RT
#module load python/3.7
#module load scipy-stack
#source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index
#pip install --no-index --upgrade pip
#pip install --no-index -r requirements.txt
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"

python3 config.py --num_runs_start $SLURM_ARRAY_TASK_ID


#!/bin/bash
#SBATCH --account=def-jag
#SBATCH --time=00:15:00
#SBATCH --ntasks=6
#SBATCH --mem-per-cpu=512M
#SBATCH --job-name=test
#SBATCH --output=test-%J.out
#SBATCH --array=0-3

cd /home/kerrick/projects/def-jag/kerrick
module load StdEnv/2018 python/3
source tensorflow/bin/activate
module load mpi4py/3.0.3
module load scipy-stack
start_time=`date +%s`
echo "starting training..."
echo "Starting task $SLURM_ARRAY_TASK_ID"
python3 config.py --num_runs_start $SLURM_ARRAY_TASK_ID
end_time=`date +%s`
runtime=$((end_time-start_time))

echo "run time"
echo $runtime