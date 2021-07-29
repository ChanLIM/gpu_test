VRAM=11
NUM_GPUS=2


# This is for tensorflow
echo *********************************************************************
echo START TENSORFLOW GPU TEST 
python3 GPU_stress_test_in_tensorflow/main.py --vram $VRAM --num_gpus $NUM_GPUS

#This is for pytorch
echo *********************************************************************
echo *********************************************************************
echo START PYTORCH GPU TEST
python3 GPU_stress_test_in_pytorch/main.py --vram $VRAM --num_gpus $NUM_GPUS
echo *********************************************************************

#NVLink test using Pytorch
echo *********************************************************************
echo START NVLINK TEST
python3 GPU_test_NVLINK/main.py --vram $VRAM --num_gpus $NUM_GPUS
