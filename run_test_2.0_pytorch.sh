VRAM=11
NUM_GPUS=2


#This is for pytorch
echo *********************************************************************
echo *********************************************************************
echo START PYTORCH GPU TEST
python3 GPU_stress_test_in_pytorch/main.py --vram $VRAM --num_gpus $NUM_GPUS
echo *********************************************************************

