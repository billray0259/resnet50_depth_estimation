# How to run
pip install the requirements.txt file

`pip install -r requirements.txt`

Download the labeled dataset from this page: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

and put it in `data/nyu_depth_v2_labeled.mat`

Clone this repository: https://github.com/AndrewAtanov/simclr-pytorch

add an __init__.py file in the main direcotry of that repository

In simclr-pytorch/models/resnet.py modify the code at line 49 to:
```
if not self.hparams.return_feature_map:
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
```

Download their pretrained models and make sure there is a .pth file at `pretrained_models/resnet50_imagenet_bs2k_epochs600.pth.tar`

In simclr-pytorch/models/ssl.py make the following adjustments
```
# add to line 297
parser.add_argument('--return_feature_map', default=False, type=bool, help='If the 7x7x2048 feature map is returned from the encoder')
```

```
# add to line 310
ckpt['hparams'].return_feature_map = hparams.return_feature_map
```

You may also have to git clone this repository https://github.com/GabrielMajeri/nyuv2-python-toolbox , but I'm not sure if I ever use it. If it complains you can probably comment out the imports.

# How to train
Run the `training_commands.sh` file - This requires like 10GB of VRAM
Reduce the batch size if that is inaccessible and you just want to see it run.

# How to test
Run the notebooks in src/testing

# How to debug
Send me a text at 303-667-3042 (I typically don't answer unknown numbers but I would call back if I got a text) or email wray@umass.edu and I'll get back to you as soon as possible

I think this should work just fine for you but deveopment environments can be tricky and I'd be happy to help get my project working for you. 

