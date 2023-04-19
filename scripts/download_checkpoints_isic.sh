echo "Download Model Checkpoints."
mkdir checkpoints
cd checkpoints

echo "Download VGG"
wget https://datacloud.hhi.fraunhofer.de/s/Tq2fGgtEZ2mMFQ4/download/checkpoint_vgg16_isic_last.pth

echo "Download ResNet18"
wget https://datacloud.hhi.fraunhofer.de/s/pijw3KFNdcZNi2e/download/checkpoint_resnet18_isic_last.pth

echo "Download EfficientNet B0"
wget https://datacloud.hhi.fraunhofer.de/s/3WoL9Ge38jJ93xN/download/checkpoint_efficientnet_b0_isic_last.pth

cd ..