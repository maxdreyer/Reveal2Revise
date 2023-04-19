echo "Download Model Checkpoints."
mkdir checkpoints
cd checkpoints

# models train/val/test split
wget https://datacloud.hhi.fraunhofer.de/s/zTzGNdi5cfGFbwX/download/checkpoint_vgg16_bone_attacked.pth
wget https://datacloud.hhi.fraunhofer.de/s/oNt6jqPjprn7FMJ/download/checkpoint_resnet18_bone_attacked.pth
wget https://datacloud.hhi.fraunhofer.de/s/iPwdLieRJxcdnJJ/download/checkpoint_efficientnet_b0_bone_attacked.pth
