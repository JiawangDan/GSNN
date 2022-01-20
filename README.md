## Abstract

## Requirements
  ```
  python 3.7
  torch==1.6.0+cu101
  torchvision==0.7.0+cu101
  pytorch-lightning==1.4.6
  transformers
  scipy
  pandas
  scikit-learn
  ```

## Dataset
Unzip and rename files:
  ```
  mv input_A_initial.csv input_A.csv
  mv input_B_initial.csv input_B.csv
  mv input_A.csv(intermediate or final test) input_A_new.csv
  mv input_A.csv(intermediate or final test) input_A_new.csv
  ```
Move files to `data_set` directory as below:
  ```
  data_set
  └── edge_type_features.csv
      edges_train_A.csv
      edges_train_B.csv
      node_features.csv
      input_A.csv
      input_A_new.csv
      input_B.csv
      input_B_new.csv
  GSNN
  └── train.py
      validate.py
      predict.py
      models.py
      build_dataset_seq.py
      build_dataset_graph.py
      ...
  ```

## Preprocessing
In order to get the sequence and subgraph of each driving event, it is necessary to build the indexes.
  ```shell
  $ cd GSNN
  $ python build_dataset_seq.py config_a
  $ python build_dataset_graph.py config_a
  $ python build_dataset_seq.py config_b
  $ python build_dataset_graph.py config_b
  ```

## Training
Here is a train script.
  ```shell
  $ python train.py config_a
  $ python trian.py config_b
  ```
After training your model, the checkpoints will save in 'lightning_logs/logs.csv/default' directory.

## Validation
Here is a script to verify the effect of the model.
  ```shell
  $ python validate.py config_a lightning_logs/logs.csv/default/version_xx/checkpoints/last.ckpt
  $ python validate.py config_b lightning_logs/logs.csv/default/version_xx/checkpoints/last.ckpt
  ```

## Inference
Here is a script that uses the model to predict the results.
  ```shell
  $ python predict.py config_a lightning_logs/logs.csv/default/version_xx/checkpoints/last.ckpt output_A.csv
  $ python predict.py config_b lightning_logs/logs.csv/default/version_xx/checkpoints/last.ckpt output_B.csv
  ```