VRAM=11
NUM_GPUS=2


# This is for tensorflow
echo *********************************************************************
echo START TENSORFLOW 2.0 GPU TEST 
python GPU_test_tf_2.0/main.py --vram $VRAM --num_gpus $NUM_GPUS
