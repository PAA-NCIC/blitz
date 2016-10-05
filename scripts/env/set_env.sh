#OPENMP
#batch parallel max 16 threads
export OMP_NUM_THREADS=16
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_DYNAMIC="FALSE" 
#Batch level
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
