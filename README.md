# SleepEGAN

Deep neural networks have played an important role in the automatic classification of sleep stages due to their strong representation and in-model feature transformation abilities. However,  class imbalance and individual heterogeneity which typically exist in raw EEG signals of sleep data can significantly affect the classification performance of any machine learning algorithms. To solve these two problems, this paper develops a generative adversarial network (GAN)-powered ensemble deep learning model, named SleepEGAN, for the imbalanced classification of sleep stages. 
To alleviate class imbalance, we propose a new GAN (called EGAN) architecture adapted to the features of EEG signals for data augmentation. The generated samples for minority classes are used in the training process. In addition, we design a cost-free ensemble learning strategy to reduce the model estimation variance caused by the heterogeneity between the validation and test sets, to enhance the accuracy and robustness of prediction performance. \textcolor{red}{We show that the proposed method improves classification accuracy compared to several existing state-of-the-art methods. 
The overall classification accuracy and macro F1-score obtained by our SleepEGAN method on three public sleep datasets are: Sleep-EDF-39: 86.8% and 81.9%; Sleep-EDF-153: 83.8% and 78.7%; SHHS: 88.0% and 82.1%.}

Cheng X, Huang K, Zou Y, et al. SleepEGAN: A GAN-enhanced Ensemble Deep Learning Model for Imbalanced Classification of Sleep Stages[J]. arXiv preprint arXiv:2307.05362, 2023.


## Model Architecture
Note: Fs is the sampling rate of the input EEG signals

## Performance Comparison
Note: ACC = accuracy, MF1 = Macro F1-Score


## Environment

* CUDA 10.0
* cuDNN 7
* Tensorflow 1.13.1
* pytorch  2.1.0

## Create a virtual environment with conda

```bash
conda create -n SleepEAGN python=3.6
conda activate SleepEAGN
pip install -r requirements.txt
```

## How to run

1. `python download_sleepedf.py`
1. `python prepare_sleepedf.py`
1. `python trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19`
1. `python predict.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best`

## Citation
If you find this useful, please cite our work as follows:

@article{cheng2023sleepegan,
  title={SleepEGAN: A GAN-enhanced Ensemble Deep Learning Model for Imbalanced Classification of Sleep Stages},
  author={Cheng, Xuewei and Huang, Ke and Zou, Yi and Ma, Shujie},
  journal={arXiv preprint arXiv:2307.05362},
  year={2023}
}

## Licence
- For academic and non-commercial use only
- Apache License 2.0
