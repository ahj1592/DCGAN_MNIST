# DCGAN_MNIST

## how to use
```bash
$ python train.py --help
usage: train.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS] [--save_dir SAVE_DIR] [--save_step SAVE_STEP]
                [--noise_dim NOISE_DIM]

Set the hyperparameters of DCGANs

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        The Batch Size used in dataset (default: 128)
  --lr LR, -l LR        Learning rate of DCGANs. recommend 0.0002 (default: 0.0002)
  --epochs EPOCHS, -e EPOCHS
                        The number of epochs to train (default: 100)
  --save_dir SAVE_DIR, -d SAVE_DIR
                        The images directory where to save. Do not use [data], [utils] for directory name. (default:
                        results)
  --save_step SAVE_STEP, -s SAVE_STEP
                        Interval of saving image (default: 500)
  --noise_dim NOISE_DIM, -z NOISE_DIM
                        Dimension of noise vector Z (default: 64)
$
```
