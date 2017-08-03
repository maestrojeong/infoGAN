# Infogan Implementation
Infogan with 3 different models  
* wgan_gp(gradient_penalty)
* wgan(weight clipping)
* gan

## Requirement
tensorflow==1.2.1
matplotlib==2.0.2

## Files
1. **main.py** - infoGAN train
2. **ops.py** - basic operations based on tensorflow
3. **utils.py** - basic operations not based on tensorflow
4. **nets.py** - 3 basic nets of infoGAN(generator, discriminator, classifier)
5. **config.py** - basic configuration for **main.py**

## Run
```bash
python main.py
```

## dataset
MNIST
