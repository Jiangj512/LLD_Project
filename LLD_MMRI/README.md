# This folder stores the source code for training and testing

## Directory structure：
```
├── LLD_MMRI
    ├── data
        ├── dataset_net1.py
        ├── dataset_net2.py
    ├── lld
        ├── lld_mmri
            ├── train_net1.txt
            ├── train_net2.txt
            ├── val_net1.txt
            ├── val_net2.txt
    ├── model
        ├── net1
            ├── log
            ├── log_net1.txt
            ├── epoch_499_acc_0.75.pth
        ├── net2
            ├── log
            ├── log_net2.txt
            ├── epoch_495_acc_0.7670940170940171.pth
    ├── net
        ├── net1.py
        ├── net2.py
    ···
```

## Training：

We use two networks and two data divisions for training.
Running **train_net1.py** and **train_net2.py** respectively will get the training data for **net1** and **net2** in the "model" folder.
Finally, we reserve the model weights in each folder with the highest accuracy on the validation set for testing.

## Testing：

Run **test.py** to test and get the predicted data. This will generate a **MediSegLearner.json**.

## Download: 
**Due to space limitations, please visit our web site [MediSegLearner](https://pan.baidu.com/s/1UFbIR2PZJh4Fxb2DnUOLHA?pwd=n31z) to download the trained model weights(path: MediSegLearner/LLD_MMRI/model) separately.**
