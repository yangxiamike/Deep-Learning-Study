import keras
import numpy as np
import scipy
import datetime
from data_load.Data_loader import Data_loader


def train(epochs,batch_size =1,sample_interval=50):
    start_time = datetime.datetime.now()
    data_load = Data_loader()


    for epoch in range(epochs):
        for batch_i , (imgs_A,imgs_B) in enumerate(data_load.load_batch(training_num=1000,batch_size=32,is_training=True)):
            a = keras.Model.train_on_batch(x,y )

            elapsed_time = datetime.datetime.now()-start_time



            if batch_i % sample_interval(50) == 0:
                sample_images()


def sample_images(self, epoch, batch_i):
    os.makedirs('%s' % 'sample_image', exist_ok=True)

    imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Demo (for GIF)
    # imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
    # imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

    # Translate images to the other domain
    fake_B = self.g_AB.predict(imgs_A)
    fake_A = self.g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = self.g_BA.predict(fake_B)
    reconstr_B = self.g_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
    plt.close()
