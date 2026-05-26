# ReLiB-100k

## Dataset

Currently, only a demo dataset is provided, containing charge and discharge data for 20 individual 15.5 Ah battery cells. The complete raw_data dataset, comprising 106,693 battery cells, will be released publicly on Kaggle after the paper is accepted, at https://www.kaggle.com/datasets/liuyunlong0621/relib-100k.
**Update (2026-05-26): The complete dataset has been uploaded publicly.**

------

## Code

The functions of the executable scripts are as follows:

- **1_2_generate_dataset.py**
   The complete dataset is divided into training, validation, and test sets in an 8:1:1 ratio based on random seeds, and each charging curve is downsampled to the same length of 512.
- **3_1_contrastive_pre_train.py**
   Performs contrastive learning pre-training for the CapCLR model.
- **3_2_finetune.py**
   Conducts end-to-end fine-tuning of the CapCLR model.
- **3_2_finetune_run.py**
   A batch execution script for CapCLR fine-tuning experiments.
- **3_3_benchmark.py**
   Runs benchmark testing of 10 baseline models.
- **3_3_benchmark_run.py**
   A batch execution script for model benchmark experiments.
- **4_1_results2xlsx.py**
   Aggregates experiment results and exports them to Excel format.




