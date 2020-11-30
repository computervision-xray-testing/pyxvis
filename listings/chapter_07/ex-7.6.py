from pyxvis.learning.gan import DCGAN

gan_proc = 0 # 0 training , # 1 testing

# Training
if gan_proc == 0: # Training
    path_file = '../data/shuriken_32x32.npy' # file of real patches
    epochs    = 15000                         # number of epochs
    interval  = 250                          # saving intervals
    dcgan     = DCGAN(path_file)
    dcgan.train(epochs=epochs, batch_size=32, save_interval=interval)

else:  # Testing (one generation of simulated images)
    size = 32              # size of the image, eg. 32 for 32x32 pixels
    # trained model h5 file
    gan_weights_file = '../output/GAN/models/gan_model_015000.h5'  
    N = 200                # number of synthetic images to be generated
    dcgan = DCGAN(size)
    dcgan.load_gan_model(gan_weights_file)
    dcgan.save_gan_examples()
    dcgan.save_synthetic_images('output',N)
