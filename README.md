
# Creating MNIST using GANs
The goal of this project is to create 'MNIST' digits using a Wasserstein GAN (generative adversarial network).
### Layout of project:
1) **Train.py:** File to initiate training
2) **data.py:** Data generator is contained here
3) **GAN.py:** Contains code for model and training process 
4) **Inference.py:** Creates visuals for trained model
5) **Util.py:** Holds misc. functions
6) **params.py:** Holds the parameters for the project
7) **plot_loss.py:** Creates a quick visual for losses

### Prerequisites
 1) Tensorflow
 3) Numpy
 4) Matplotlib
 5) OpenCV
 6) TQDM
 7) Pandas
 
### Usage
Training of the model involves running _train.py_ then _inference.py_ to create the visuals

### Examples of what the model can achieve:
![image](./imgs/14.png =456x456) 

### An Example of an interpolation through latent space of the trained model:
![alt text](./imgs/animation.gif =456x456)

