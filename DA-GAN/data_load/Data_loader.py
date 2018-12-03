import numpy as np
from glob import glob
import scipy
import os
import imageio

class Data_loader(object):
    def __init__(self):
        '''
        :param is_training: True
        '''
        self.source = '/Users/Mike/Documents/machine learning/deep learning/da-gan/CUB_200_2011/'
        #self.is_training = is_training
        self.dictionary = self.file_dictionary()
        self.num_species = 200

    """
    def split_train_test(self):
        '''
        :param is_training:
        :return: choice: index of training images
                type = lise
        '''
        path = self.source+'CUB_200_2011/CUB_200_2011/'
        with open(path+'train_test_split.txt', 'r') as file:
            choice = []
            index = []
            lines = file.readlines()
            for line in lines:
                line = line[:-1]
                a,b = [i for i in line.split(' ')]
                index.append(int(b))
                choice.append(int(a))
            file.close()
        if self.is_training == True:
            index = np.where(np.array(index)>0)
            choice = np.array(choice)
            choice = choice[index[0]]
        else:
            index = np.where(np.array(index)==0)
            choice = np.array(choice)
            choice = choice[index[0]]
        return list(choice)
    """
    def file_dictionary(self):
        """
        :param is_training:
        :return: dictionary
        dictionary = {
                    'species_name':'',
                    'index':[],
                    'path_name':[]
            }
        type is training or test
        """
        path = ('/').join(['CUB_200_2011','images.txt'])
        path = self.source+path
        dictionary = {}

        for i in range(1,201):
            dictionary[i] = {
                    'species_name':'',
                    'index':[],
                    'path_name':[]
            }

        with open(path,'r') as file:
            path = ('/').join(['CUB_200_2011','images'])
            lines = file.readlines()
            for line in lines:
                line = line[:-1]
                index,name =[i for i in line.split(' ')]
                index = int(index)
                species,path_name = [i for i in name.split('/')]
                species_index,species_name = [i for i in species.split('.')]

                path_name = ('/').join([path,(str(species_index)+'.'+species_name),path_name])
                species_index = int(species_index)
                dictionary[species_index]['species_name'] = species_name
                dictionary[species_index]['index'].append(index)
                dictionary[species_index]['path_name'].append(path_name)
        file.close()
        return dictionary

    def load_batch(self,training_num=1000,batch_size = 1, is_training = True):
        '''
        :param batch_size: int
        :param is_training: True
               dictionary = {
                    'species_name':'',
                    'index':[],
                    'path_name':[]
            }
        :yield generator
        max batch = 41

        get path = self.source + dictionary[index]['path_name'][img_index]
        '''
        self.train_num = training_num
        ls = np.arange(1,self.num_species+1)
        for i in range(self.train_num):
            A_index,B_index = np.random.choice(ls,size=2,replace=False)
            path_A = self.dictionary[A_index]['path_name']
            path_B = self.dictionary[B_index]['path_name']
            path_A = np.random.choice(path_A,size = batch_size,replace=False)
            path_B = np.random.choice(path_B,size = batch_size,replace=False)
            imgs_A, imgs_B = [],[]
            for img_A,img_B in zip(path_A,path_B):
                img_A = self.source+img_A
                img_B = self.source+img_B
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)
                if np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.array(imgs_A)/127.5-1
            imgs_B = np.array(imgs_B)/127.5-1

            yield imgs_A,imgs_B


    def load_data(self,domain = 'A',batch_size = 1,is_Testing = True):
        return 0


    def load_img(self,path):
        img = self.imread(path)
        img = img/127.5-1
        return img[np.newaxis,:,:,:]    

    def imread(self, path):
        return imageio.imread(path).astype(np.float)
        #return scipy.misc.imread(path,mode='RGB').astype(np.float)


if __name__ == '__main__':
    load = Data_loader()





