import os, sys
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import pickle as pk
import shutil
import os.path as osp


class CubBirds(object):

    def __init__(self, root):
        self.file_dict = {'Cid':'classes.txt',
                          'imageCid':'image_class_labels.txt',
                          'imageId':'images.txt',
                          'imageTVT':'train_test_split.txt'
                          }
        self.root = root

    def _className(self):
        with open(os.path.join(self.root, self.file_dict['Cid'])) as f:
            coxt = f.readlines()
            class_names = [x.split()[-1] for x in coxt]
        self.class_names = class_names
        return self.class_names

    def _imdb(self):

        with open(os.path.join(self.root, self.file_dict['imageId'])) as f:
            coxt = f.readlines()
            imageId = [int(x.split()[0]) for x in coxt]
            imagePath = [x.split()[-1] for x in coxt]
        df = {'imageId':imageId, 'imagePath':imagePath}
        imdb = pd.DataFrame(data=df)
        imageCid = [x.split('/')[0] for x in imdb['imagePath']]
        with open(os.path.join(self.root, self.file_dict['imageTVT'])) as f:
            coxt = f.readlines()
            imageTVT = [int(x.split()[-1]) for x in coxt]
        imdb['imageCid'] = imageCid
        imdb['imageTVT'] = imageTVT

        return imdb


class StCars(object):

    def __init__(self, root):
        with open(osp.join(root, 'imdb.pkl'), 'rb') as handle:
            annos = pk.load(handle)
        self.classnames = annos['classnames']
        self.classdict = annos['classdict']
        self.testdata = annos['annos_test']
        self.traindata = annos['annos_train']
        self.root = root

    def _createTVTFolders(self):
        if not osp.exists(osp.join(self.root, 'train', self.classnames[0])):
            for classname in self.classnames:
                os.makedirs(osp.join(self.root, 'train', classname))
                os.makedirs(osp.join(self.root, 'val', classname))
                os.makedirs(osp.join(self.root, 'trainval', classname))
                os.makedirs(osp.join(self.root, 'test', classname))


class StDogs(object):

    def __init__(self, root):
        self.root = root
        with open(osp.join(root, 'imdb.pkl'), 'rb') as handle:
            annos = pk.load(handle)

        self.classdict = annos['classdict']
        self.classnames = annos['classnames']
        self.traindata = annos['traindata']
        self.trainlabels = annos['trainlabels']
        self.testdata = annos['testdata']
        self.testlabels = annos['testlabels']

    def _createTVTFolders(self):
        if not osp.exists(osp.join(self.root, 'train', self.classnames[0])):
            for classname in self.classnames:
                os.makedirs(osp.join(self.root, 'train', classname))
                os.makedirs(osp.join(self.root, 'val', classname))
                os.makedirs(osp.join(self.root, 'trainval', classname))
                os.makedirs(osp.join(self.root, 'test', classname))


class VggAircraft(object):

    def __init__(self, root):
        with open(os.path.join(root, 'imdb.pkl'), 'rb') as handle:
            annos = pk.load(handle)

        self.classdict_variant = annos['classdict_variant']
        self.classdict_family = annos['classdict_variant']
        self.classdict_manufacturer = annos['classdict_variant']
        self.classnames_variant = annos['classnames_variant']
        self.classnames_family = annos['classnames_family']
        self.classnames_manufacturer = annos['classnames_manufacturer']
        self.traindata = annos['traindata']
        self.trainvaldata = annos['trainvaldata']
        self.testdata = annos['testdata']
        self.valdata = annos['valdata']
        self.root = root

    def _createTVTFolders(self):
        if not osp.exists(osp.join(self.root, 'train', self.classnames_variant[0])):
            for classname in self.classnames_variant:
                os.makedirs(osp.join(self.root, 'train', classname))
                os.makedirs(osp.join(self.root, 'val', classname))
                os.makedirs(osp.join(self.root, 'trainval', classname))
                os.makedirs(osp.join(self.root, 'test', classname))


class NaBirds(object):
    def __init__(self, root):
        with open(os.path.join(root, 'imdb.pkl'), 'rb') as handle:
            annos = pk.load(handle)

        self.prntclassid = annos['prntclassid']
        self.classnames = annos['classnames']
        self.subclassid = annos['subclassid']
        self.testdata = annos['testdata']
        self.traindata = annos['traindata']
        self.root = root

    def _createTVTFolders(self):
        subclassid = list(set(self.subclassid.values()))
        if not osp.exists(osp.join(self.root, 'train', subclassid[0])):
            for classname in subclassid:
                os.makedirs(osp.join(self.root, 'train', classname))
                os.makedirs(osp.join(self.root, 'val', classname))
                os.makedirs(osp.join(self.root, 'trainval', classname))
                os.makedirs(osp.join(self.root, 'test', classname))


class WdDogs(object):

    def __init__(self, root):
        with open(os.path.join(root, 'imdb.pkl'), 'rb') as handle:
            annos = pk.load(handle)

        self.classdict = annos['classdict']
        self.classnames = annos['classnames']
        self.traindata = annos['traindata']
        self.trainvaldata = annos['trainvaldata']
        self.testdata = annos['testdata']
        self.valdata = annos['valdata']
        self.root = root

    def _createTVTFolders(self):
        if not osp.exists(osp.join(self.root, 'train', self.classnames[0])):
            for classname in self.classnames:
                os.makedirs(osp.join(self.root, 'train', classname))
                os.makedirs(osp.join(self.root, 'val', classname))
                os.makedirs(osp.join(self.root, 'trainval', classname))
                os.makedirs(osp.join(self.root, 'test', classname))

def creatDataset(root, datasetname=None):

    if datasetname is not None:
        trainpath = os.path.join(root, 'train')
        valpath = os.path.join(root, 'val')
        testpath = os.path.join(root, 'test')
        trainvalpath = os.path.join(root, 'trainval')

        if not os.path.exists(trainpath):
            os.makedirs(trainpath)
            os.makedirs(valpath)
            os.makedirs(testpath)
            os.makedirs(trainvalpath)

        # checking the train/val/test integrity
        train_folders = os.listdir(trainpath)
        val_folders = os.listdir(valpath)
        test_folders = os.listdir(testpath)
        trainval_folders = os.listdir(trainvalpath)
        assert len(train_folders) == len(val_folders) == len(
            test_folders), "The train/val/test datasets are not complete"
        num_train_data = sum([len(os.listdir(os.path.join(trainpath, subfolder))) for subfolder in train_folders])
        num_val_data = sum([len(os.listdir(os.path.join(valpath, subfolder))) for subfolder in val_folders])
        num_test_data = sum([len(os.listdir(os.path.join(testpath, subfolder))) for subfolder in test_folders])
        num_trainval_data = sum(
            [len(os.listdir(os.path.join(trainvalpath, subfolder))) for subfolder in trainval_folders])

        if datasetname is "cubbirds":

            if num_test_data+num_train_data+num_val_data == 11788 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True

            # if the train/val/test datasets are not exist
            birds = CubBirds(root)
            class_names = birds._className()

            if os.path.exists(os.path.join(root, 'imdb.pkl')):
                with open(os.path.join(root, 'imdb.pkl'),'rb') as f:
                    pdata = pk.load(f)
                    train_pd, val_pd,test_pd,  = pdata['train'], pdata['val'], pdata['test']
            else:
                imdb = birds._imdb()

                test_pd = imdb.loc[imdb['imageTVT']==0]
                train_tpd = imdb.loc[imdb['imageTVT'] == 1]

                train_pd = pd.DataFrame()
                val_pd = pd.DataFrame()

                for class_name in class_names:
                    trainval = train_tpd.loc[train_tpd['imageCid'] == class_name]
                    permuind = np.random.permutation(trainval.index)
                    # print(permuind)
                    train_pd = train_pd.append(trainval.loc[permuind[:-3]], ignore_index=True)
                    val_pd = val_pd.append(trainval.loc[permuind[-3:]], ignore_index=True)
                    # print(trainval.index)
                with open(os.path.join(root, 'imdb.pkl'),'wb') as f:
                    pk.dump({'train':train_pd, 'val':val_pd, 'test':test_pd},f)

            for class_name in class_names:

                if not os.path.exists(os.path.join(trainvalpath, class_name)):
                    os.mkdir(os.path.join(trainpath, class_name))
                    os.mkdir(os.path.join(valpath, class_name))
                    os.mkdir(os.path.join(testpath, class_name))
                    os.mkdir(os.path.join(trainvalpath, class_name))


                train_dst_path = os.path.join(trainpath, class_name)
                val_dst_path = os.path.join(valpath, class_name)
                test_dst_path = os.path.join(testpath, class_name)
                trainval_dst_path = os.path.join(trainvalpath, class_name)

                newtrainpd = train_pd.loc[train_pd['imageCid'] == class_name]
                newtrainpd_index = newtrainpd.index
                for i_ in newtrainpd_index:
                    src_path = os.path.join(root, 'images', newtrainpd.loc[i_,'imagePath'])
                    shutil.copy(src_path, train_dst_path)
                    shutil.copy(src_path, trainval_dst_path)

                newvalpd = val_pd.loc[val_pd['imageCid'] == class_name]
                newvalpd_index = newvalpd.index
                for i_ in newvalpd_index:
                    src_path = os.path.join(root, 'images', newvalpd.loc[i_,'imagePath'])
                    shutil.copy(src_path, val_dst_path)
                    shutil.copy(src_path, trainval_dst_path)

                newtestpd = test_pd.loc[test_pd['imageCid'] == class_name]
                newtestpd_index = newtestpd.index
                for i_ in newtestpd.index:
                    print(i_)
                    src_path = os.path.join(root, 'images', newtestpd.loc[i_,'imagePath'])
                    shutil.copy(src_path, test_dst_path)
            print("Successfully creating train/val/test sets.")
            return True
        elif datasetname is "stcars":

            if num_test_data+num_train_data+num_val_data == 16185 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True

            # if the train/val/test datasets are not exist
            cars = StCars(root)
            class_names = cars.classnames
            class_dict = cars.classdict
            traindata = cars.traindata
            testdata = cars.testdata

            cars._createTVTFolders()

            for line in traindata:
                train_src_path = osp.join(cars.root, line['relative_im_path'])
                class_name = class_dict[line['class']]
                trainval_dst_path = osp.join(trainvalpath, class_name)
                shutil.copy(train_src_path, trainval_dst_path)
            for line in testdata:
                test_src_path = osp.join(cars.root, line['relative_im_path'])
                class_name = class_dict[line['class']]
                test_dst_path = osp.join(testpath, class_name)
                shutil.copy(test_src_path, test_dst_path)

            # build train and validation sets from the trainval set
            subfolders = os.listdir(trainvalpath)
            for subfolder in subfolders:
                imgs = os.listdir(osp.join(trainvalpath, subfolder))
                num_imgs = len(imgs)
                rndidx = np.random.permutation(num_imgs)
                num_val = int(np.floor(0.1 * num_imgs))
                num_train = num_imgs - num_val
                train_dst_path = osp.join(trainpath, subfolder)
                val_dst_path = osp.join(valpath, subfolder)
                for idx in rndidx[:num_train]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), train_dst_path)
                for idx in rndidx[num_train:]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), val_dst_path)

            print("Successfully creating train/val/test sets.")
            return True
        elif datasetname is 'stdogs':

            if num_test_data+num_train_data+num_val_data == 20580 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True

            # if the train/val/test datasets are not exist
            dogs = StDogs(root)
            class_names = dogs.classnames
            class_dict = dogs.classdict
            train_data = dogs.traindata
            train_labels = dogs.trainlabels
            test_data = dogs.testdata
            test_labels = dogs.testlabels

            dogs._createTVTFolders()

            for imgpath in train_data:
                class_name = imgpath.split('/')[0]
                train_src_path = osp.join(dogs.root, 'Images', imgpath)
                trainval_dst_path = osp.join(trainvalpath, class_name)
                shutil.copy(train_src_path, trainval_dst_path)
            for imgpath in test_data:
                class_name = imgpath.split('/')[0]
                test_src_path = osp.join(dogs.root, 'Images', imgpath)
                test_dst_path = osp.join(testpath, class_name)
                shutil.copy(test_src_path, test_dst_path)

            # build train and validation sets from the trainval set
            subfolders = os.listdir(trainvalpath)
            for subfolder in subfolders:
                imgs = os.listdir(osp.join(trainvalpath, subfolder))
                num_imgs = len(imgs)
                rndidx = np.random.permutation(num_imgs)
                num_val = int(np.floor(0.1 * num_imgs))
                num_train = num_imgs - num_val
                train_dst_path = osp.join(trainpath, subfolder)
                val_dst_path = osp.join(valpath, subfolder)
                for idx in rndidx[:num_train]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), train_dst_path)
                for idx in rndidx[num_train:]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), val_dst_path)

            print("Successfully creating train/val/test sets.")
            return True
        elif datasetname is "vggaircraft":
            if num_test_data+num_train_data+num_val_data == 10000 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True
            aircrafts = VggAircraft(root)
            traindata = aircrafts.traindata
            trainvaldata = aircrafts.trainvaldata
            valdata = aircrafts.valdata
            testdata = aircrafts.testdata
            classnames = aircrafts.classnames_variant
            classdict = aircrafts.classdict_variant
            aircrafts._createTVTFolders()

            for row in traindata:
                img_src_path = osp.join(root, 'images', row[0]+'.jpg')
                img_dst_path = osp.join(root, 'train', row[1])
                shutil.copy(img_src_path, img_dst_path)

            for row in trainvaldata:
                img_src_path = osp.join(root, 'images', row[0]+'.jpg')
                img_dst_path = osp.join(root, 'trainval', row[1])
                shutil.copy(img_src_path, img_dst_path)

            for row in valdata:
                img_src_path = osp.join(root, 'images', row[0]+'.jpg')
                img_dst_path = osp.join(root, 'val', row[1])
                shutil.copy(img_src_path, img_dst_path)

            for row in testdata:
                img_src_path = osp.join(root, 'images', row[0]+'.jpg')
                img_dst_path = osp.join(root, 'test', row[1])
                shutil.copy(img_src_path, img_dst_path)

            print("Successfully creating train/val/test sets.")
            return True
        elif datasetname is "nabirds":
            if num_test_data+num_train_data+num_val_data == 48562 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True
            nabirds = NaBirds(root)
            traindata = nabirds.traindata
            testdata = nabirds.testdata
            classnames = nabirds.classnames
            prntclassid = nabirds.prntclassid
            subclassid = nabirds.subclassid
            nabirds._createTVTFolders()

            # trainval data
            for row in traindata:
                img_src_path = osp.join(root, row[0])
                img_dst_path = osp.join(root, 'trainval', row[1])
                shutil.copy(img_src_path, img_dst_path)

            # testing data
            for row in testdata:
                img_src_path = osp.join(root, row[0])
                img_dst_path = osp.join(root, 'test', row[1])
                shutil.copy(img_src_path, img_dst_path)

            # build train and validation sets from the trainval set
            subfolders = os.listdir(trainvalpath)
            for subfolder in subfolders:
                imgs = os.listdir(osp.join(trainvalpath, subfolder))
                num_imgs = len(imgs)
                rndidx = np.random.permutation(num_imgs)
                num_val = int(np.floor(0.1 * num_imgs))
                num_train = num_imgs - num_val
                train_dst_path = osp.join(trainpath, subfolder)
                val_dst_path = osp.join(valpath, subfolder)
                for idx in rndidx[:num_train]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), train_dst_path)
                for idx in rndidx[num_train:]:
                    shutil.copy(osp.join(trainvalpath, subfolder, imgs[idx]), val_dst_path)

            print("Successfully creating train/val/test sets.")
            return True
        elif datasetname is "wddogs":
            if num_test_data+num_train_data+num_val_data == 299458 and num_train_data+num_val_data == num_trainval_data:
                print("train/val/test sets are already exist.")
                return True
            wddogs = WdDogs(root)
            traindata = wddogs.traindata
            testdata = wddogs.testdata
            valdata = wddogs.valdata
            trainvaldata = wddogs.trainvaldata
            classnames = wddogs.classnames
            classdict = wddogs.classdict
            wddogs._createTVTFolders()

            for elmt in traindata:
                img_src_path = osp.join(root, elmt['imgname'])
                img_dst_path = osp.join(trainpath, elmt['imgCname'])
                shutil.copy(img_src_path, img_dst_path)

            for elmt in valdata:
                img_src_path = osp.join(root, elmt['imgname'])
                img_dst_path = osp.join(valpath, elmt['imgCname'])
                shutil.copy(img_src_path, img_dst_path)

            for elmt in trainvaldata:
                img_src_path = osp.join(root, elmt['imgname'])
                img_dst_path = osp.join(trainvalpath, elmt['imgCname'])
                shutil.copy(img_src_path, img_dst_path)

            for elmt in testdata:
                img_src_path = osp.join(root, elmt['imgname'])
                img_dst_path = osp.join(testpath, elmt['imgCname'])
                shutil.copy(img_src_path, img_dst_path)

            print("Successfully creating train/val/test sets.")
            return True
        else:
            print("This dataset has not been implemented.")
            return False

    else:
        print("You should provide the dataset name for proceeding.\n")
        return False


if __name__ == "__main__":
    pass
