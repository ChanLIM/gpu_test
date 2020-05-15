VRAM=11
NUM_GPUS=2


#This is for NVLINK, pytorch
echo '*********************************************************************'
echo '*********************************************************************'
echo "START NVLINK GPU TEST\n"

python3 GPU_test_NVLINK/main.py --vram $VRAM --num_gpus $NUM_GPUS
echo '*********************************************************************'