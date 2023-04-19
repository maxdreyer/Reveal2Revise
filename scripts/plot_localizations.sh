file_name=experiments.preprocessing.localize_artifacts

### VGG16 ISIC

config_file=config_files/correcting_isic/local/vgg16_Vanilla_sgd_lr0.0001_band_aid.yaml
python3 -m $file_name --config_file $config_file  --artifact band_aid --layer_name features.7 --save_examples True
python3 -m $file_name --config_file $config_file  --artifact ruler --layer_name features.28 --save_examples True
python3 -m $file_name --config_file $config_file  --artifact skin_marker --layer_name features.12 --save_examples True

config_file=config_files/correcting_isic/local/vgg16_Vanilla_sgd_lr0.0001_big_l.yaml
python3 -m $file_name --config_file $config_file  --artifact big_l --layer_name features.28 --save_examples True