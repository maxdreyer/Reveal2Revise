echo "Download Model Checkpoints."
mkdir checkpoints
cd checkpoints

echo "Download VGG of Iteration 1"
wget https://datacloud.hhi.fraunhofer.de/s/cPR6nRTaPKnTLLe/download/vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_skin_marker.ckpt

echo "Download VGG of Iteration 2"
wget https://datacloud.hhi.fraunhofer.de/s/CcbNxWd3Di2Apae/download/vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_skin_marker-band_aid.ckpt
cd ..