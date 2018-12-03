import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

os.chdir('/Users/Mike/Documents/machine learning/deep learning/cycle-gan/images/sample')
files = glob('*')
count = 0
for file in files:
    print(count)
    gen_imgs = np.load(file)


    titles = ['Original', 'Translated', 'Reconstructed']
    r, c = 2, 3
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            print(gen_imgs[cnt].shape)
            axs[i, j].set_title(titles[j])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("/Users/Mike/Documents/machine learning/deep learning/cycle-gan/images/generate_256/%s.jpg" % file[:-8])
    plt.close()
    count+=1

