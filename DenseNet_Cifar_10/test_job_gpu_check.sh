export CUDA_HOME=/usr/local/cuda:/usr/local/cuda-8.0:/home/jrafatiheravi/src/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

module load anaconda3
source activate jacobenv
(export CUDA_VISIBLE_DEVICES=0
python test_gpu_devices.py >> test_1.txt) &
(export CUDA_VISIBLE_DEVICES=1
python test_gpu_devices.py >> test_2.txt) &
(export CUDA_VISIBLE_DEVICES=2
python test_gpu_devices.py >> test_3.txt) &
(export CUDA_VISIBLE_DEVICES=3
python test_gpu_devices.py >> test_4.txt) &
wait
