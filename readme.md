# 3D-Aware Multi-Modal LLM

This project contains code and resources for evaluating and analyzing the performance of 3D-Aware Multi-Modal LLMs.

## 📂 Data Preparation

### Image Data (SPAR-7M)
Before proceeding, you must download the source images. Please refer to the [SPAR-7M Hugging Face repository](https://huggingface.co/datasets/kmnp/SPAR-7M) for detailed instructions.

You can download the dataset using the `huggingface-cli` or `git`:

```bash
# Example using huggingface-cli
huggingface-cli download --repo-type dataset kmnp/SPAR-7M --local-dir ./data/SPAR-7M
```

Ensure that the images are downloaded and extracted correctly before starting the training process.

### Training Data (JSON)
To prepare the training data, please download the compressed training JSON files from the Google Drive link below and unzip them into the `train/` directory.

* **Train Data Download Link:** [https://drive.google.com/file/d/1YqoMegnmXvM1pHYzaCVnB94XZIqSiQgO/view?usp=sharing]

### Test Data (JSON)
The test JSON files are located in the `test/` folder of this repository. These files are ready to be used and should be integrated into `llms-eval` during the evaluation phase.

### Plucker Ray Conversion
This project utilizes Plucker ray representations for specific modalities. To generate these visualizations from your image data, run the `convert_to_plucker.py` script. This script will process the images and save the Plucker ray visualizations in the designated directories (e.g., `plucker_c2w`).

```bash
python convert_to_plucker.py
````

*Note: Ensure your `train` and `test` JSON files are in place before running the conversion script.*

## 🚀 Training

Model training is conducted using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git). Please perform **LoRA SFT** using the default parameters provided by the framework.

You can obtain models trained on different data modalities by using the following datasets respectively:

  * `train/obj_spatial_relation_oo_mv_both_train.json`
  * `train/obj_spatial_relation_oo_mv_none_train.json`
  * `train/obj_spatial_relation_oo_mv_plucker_train.json`
  * `train/obj_spatial_relation_oo_mv_text_train.json`

> **⚠️ Important Note:**
> The image paths defined in the `*-train.json` files above must match the actual local paths where you saved the images. You may need to either modify the paths in the JSON files or adjust your directory structure (e.g., using symbolic links) to ensure the training script can locate the images correctly.

## 🧪 Evaluation

Model evaluation is performed using the [llms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval.git) framework.

1.  Add your trained **checkpoint** path and the corresponding **evaluation JSON files** (located in the `test/` folder) to the `llms-eval` configuration.
2.  Modify the evaluation script to match your configuration and run it.
      * *For detailed instructions on configuration and execution, please refer to the official `llms-eval` GitHub repository.*

Once evaluation is complete, the results can be used for the analysis step described below.

## 💾 Pre-saved Results

If you wish to skip the training and evaluation steps and proceed directly to verifying the analysis, you can download the pre-saved evaluation results via the following Google Drive link:

  * **Pre-saved Evaluation Results Download Link:** [https://drive.google.com/file/d/1zLrpDphVp_9Hv7zByWX-slyiBztMur15/view?usp=sharing]

## 📊 Running the Analysis

To perform detailed analysis and generate reports based on the downloaded (or newly generated) evaluation results, run the analysis script:

```bash
python advanced_spar_analyzer_final.py
