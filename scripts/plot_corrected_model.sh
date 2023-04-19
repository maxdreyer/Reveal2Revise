file_name=experiments.plots.plot_corrected_model

## VGG16 ISIC
for config_path in {vgg16_RRR_ExpMax_sgd_lr0.0001_lamb5000_band_aid.yaml,\
vgg16_AClarc_sgd_lr0.0001_lamb1.0_features.14_band_aid.yaml,\
vgg16_PClarc_sgd_lr0.0001_lamb1.0_features.28_band_aid.yaml,\
vgg16_CDEP_sgd_lr0.0001_lamb10_band_aid.yaml,\
vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml,\
}; do
  python3 -m $file_name --config_file config_files/correcting_isic/local/$config_path --sample_ids 531,562,560
done

## ITERATION
for config_path in {vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml,\
vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_skin_marker.yaml,\
vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_skin_marker-band_aid.yaml,\
vgg16_RRR_ExpMax_sgd_lr0.0001_lamb100_skin_marker-band_aid-ruler.yaml,\
}; do
  python3 -m $file_name --config_file config_files/correcting_isic/local/$config_path --sample_ids 1720,1683,1535,531,562,560,1446,2207,2116
done

## VGG16 Bone
for config_path in {vgg16_RRR_ExpMax_sgd_lr0.0001_lamb500_big_l.yaml,\
vgg16_AClarc_sgd_lr0.0001_lamb1.0_big_l.yaml,\
vgg16_CDEP_sgd_lr0.0001_lamb100_big_l.yaml,\
vgg16_Vanilla_sgd_lr0.0001_big_l.yaml,\
}; do
  CUDA_VISIBLE_DEVICES=0 python3 -m $file_name --config_file config_files/correcting_isic/local/$config_path --sample_ids 7416,3255,3149
done