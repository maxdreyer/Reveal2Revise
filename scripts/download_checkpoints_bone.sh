echo "Download Model Checkpoints."
mkdir checkpoints
cd checkpoints

# models train/val/test split
wget https://datacloud.hhi.fraunhofer.de/s/kMoYLaMnTzTLQ7a/download/checkpoint_vgg16_bone_last.pth
wget https://datacloud.hhi.fraunhofer.de/s/bJLiFPaEkmCFiFj/download/checkpoint_resnet18_bone_last.pth
wget https://datacloud.hhi.fraunhofer.de/s/W2weWfJ5EjL4Eit/download/checkpoint_efficientnet_b0_bone_last.pth

cd ..
