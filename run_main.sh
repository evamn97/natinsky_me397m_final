#!/bin/bash
#----------------------------------------------------
# AFM scan image analysis program to identify high-gradient feature areas.

# Last revised: 11 May 2022
# eva_mn
# 
# See .md file for notes. 
#----------------------------------------------------
#SBATCH -J natinsky_final_proj     			# Job name
#SBATCH -o natinsky_final_proj%j.out   	# Name of stdout output file(%j expands to jobId)
#SBATCH -p normal                  			# Queue (partition) name
#SBATCH -N 2                        		# Total number of nodes requested 
#SBATCH -n 128                        	# Total number of mpi tasks requested (optional) - ** set based on the dimensions of the image (e.g. 128x128 px)
#SBATCH -t 00:10:00                 		# Run time (hh:mm:ss), max is 48 hours
#SBATCH -A ME397M-DA	         			    # 'ME397M-DA' is the name of our class allocation


# **PLEASE READ**:
# conda is kind of a pain on TACC so I usually use pipenv
# if you do not have pipenv installed you can run:

# pip install --user pipenv


# once you have pipenv, you can create an env and install dependencies from requirements.txt with:

# pipenv install


module list
echo "Date: "
date
cat README.md
echo "Working directory: "
pwd

start=`date +%s`

# Launch code using pipenv virtual environment
pipenv run python3 main.py -d data

end=`date +%s`

runtime=$((end-start))
echo "Runtime: $runtime"
