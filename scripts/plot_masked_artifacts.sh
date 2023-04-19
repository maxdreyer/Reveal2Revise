file_name=experiments.plots.plot_masked_artifact

# VGG16 ISIC
config_file=config_files/correcting_isic/local/vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml
for sample_id in {531,562,560}; do
  python3 -m $file_name --config_file $config_file --sample_id $sample_id --artifact band_aid
done

for sample_id in {1555,2193,2265}; do
  python3 -m $file_name --config_file $config_file --sample_id $sample_id --artifact ruler
done

for sample_id in {1768,1375,2039}; do
  python3 -m $file_name --config_file $config_file --sample_id $sample_id --artifact skin_marker
done

# VGG16 BONE
config_file=config_files/correcting_bone/local/vgg16_Vanilla_sgd_lr0.0001_big_l.yaml
for sample_id in {7416,3255,3149}; do
  python3 -m $file_name --config_file $config_file --sample_id $sample_id --artifact big_l
done