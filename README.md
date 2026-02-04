# ReLiB-100k

## Dataset

Currently, only a demo dataset is provided, containing charge and discharge data for 20 individual 15.5Ah battery cells. The complete raw_data dataset (containing 106,693 battery cells) will be updated after the paper is accepted.

------

## Code

The functions of the executable scripts are as follows:

- **1_2_generate_dataset.py**
   Randomly samples batteries from the raw data, performs uniform downsampling, and generates the experimental dataset with an 8:1:1 split for training, validation, and testing.
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


