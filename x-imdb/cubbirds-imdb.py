import scipy.io as sio
import pickle as pk
import os, sys
import os.path as osp
import pandas as pd
import numpy as np

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)


imdbpath = osp.join(progpath, "imdb.pkl")

if not os.path.exists(imdbpath):

    file_dict = {'Cid': 'classes.txt',
                 'imageCid': 'image_class_labels.txt',
                 'imageId': 'images.txt',
                 'imageTVT': 'train_test_split.txt'
                 }

    with open(os.path.join(progpath, file_dict['Cid'])) as f:
        coxt = f.readlines()
        class_names = [x.split()[-1] for x in coxt]
    classnames = class_names


    with open(os.path.join(progpath, file_dict['imageId'])) as f:
        coxt = f.readlines()
        imageId = [int(x.split()[0]) for x in coxt]
        imagePath = [x.split()[-1] for x in coxt]
    df = {'imageId': imageId, 'imagePath': imagePath}
    imdb = pd.DataFrame(data=df)
    imageCid = [x.split('/')[0] for x in imdb['imagePath']]
    with open(os.path.join(progpath, file_dict['imageTVT'])) as f:
        coxt = f.readlines()
        imageTVT = [int(x.split()[-1]) for x in coxt]
    imdb['imageCid'] = imageCid
    imdb['imageTVT'] = imageTVT

    ####################################################33
    test_pd = imdb.loc[imdb['imageTVT'] == 0]
    train_tpd = imdb.loc[imdb['imageTVT'] == 1]

    train_pd = pd.DataFrame()
    val_pd = pd.DataFrame()

    for class_name in classnames:
        trainval = train_tpd.loc[train_tpd['imageCid'] == class_name]
        permuind = np.random.permutation(trainval.index)
        # print(permuind)
        train_pd = train_pd.append(trainval.loc[permuind[:-3]], ignore_index=True)
        val_pd = val_pd.append(trainval.loc[permuind[-3:]], ignore_index=True)
        # print(trainval.index)

    ####################### save in pickle files
    # classdict: index to class name
    # classnames: class name
    # annos_test: annotations for testing data
    # annos_train: annotations for training data

    with open(os.path.join(progpath, 'imdb.pkl'), 'wb') as f:
        pk.dump({'train': train_pd,
                 'val': val_pd,
                 'test': test_pd,
                 'classnames': classnames}, f)


else:
    with open(os.path.join(root, 'imdb.pkl'), 'rb') as f:
        pdata = pk.load(f)
        train_pd, val_pd, test_pd, classnames = pdata['train'], pdata['val'], pdata['test'], pdata['classnames']


