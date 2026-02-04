# ReLiB-100k

## Dataset

The data under the `data/generated_data` directory are obtained by uniformly sampling 3,000 batteries from each nominal capacity category and downsampling their charging curve length to 512. This dataset is used for all benchmark experiments in the paper.

Currently, only the 15.5 Ah subset is publicly available. The remaining subsets of other nominal capacities, as well as the raw data under the `data/raw_data` directory, will be released after the acceptance of the paper.

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

