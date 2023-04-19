echo "Download Model Checkpoints."
mkdir checkpoints
cd checkpoints

#MEL 0.1 (train/val/test)
wget https://datacloud.hhi.fraunhofer.de/s/pNaa9b3QpCNfSAA/download/checkpoint_vgg16_isic_attacked.pth
wget https://datacloud.hhi.fraunhofer.de/s/ZZibCgcN9r8sC3X/download/checkpoint_resnet18_isic_attacked.pth
wget https://datacloud.hhi.fraunhofer.de/s/DHdoX5XcHkjasWr/download/checkpoint_efficientnet_b0_isic_attacked.pth

cd ..