import scipy.io as sio
import pickle as pk
import os, sys

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)

imdbpath = os.path.join(parnpath, "imdb.pkl")

if not os.path.exists(imdbpath):
    annos_name = "cars_annos.mat"
    annos = sio.loadmat(os.path.join(parnpath, annos_name),
                        mat_dtype=False, squeeze_me=True, matlab_compatible=False,struct_as_record=True)

    annos_type=['relative_im_path', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'test']
    annotations = annos['annotations']
    classnames_tmp = annos['class_names'].tolist()
    classnames = list()
    for classname in classnames_tmp:
        classnames.append(classname.replace('/', ''))

    annos_train = []
    annos_test = []
    test_count = 0
    for anno in annotations:
        dl = {}
        dl['relative_im_path'] = anno[0]
        dl['bbox_x1'] = anno[1]
        dl['bbox_y1'] = anno[2]
        dl['bbox_x2'] = anno[3]
        dl['bbox_y2'] = anno[4]
        dl['class'] = anno[5] - 1
        dl['test'] = anno[-1]

        if dl['test'] == 1:
            annos_test.append(dl)
        else:
            annos_train.append(dl)

    print(len(annos_train), len(annos_test))

    classdict = {}
    for i, classname in enumerate(classnames):
        classdict[i] = classname



    ####################### save in pickle files
    # classdict: index to class name
    # classnames: class name
    # annos_test: annotations for testing data
    # annos_train: annotations for training data

    with open(os.path.join(progpath, 'imdb.pkl'), 'wb') as handle:
        pk.dump({'classdict': classdict, 'classnames': classnames,
                 'annos_test': annos_test, 'annos_train': annos_train}, handle)
else:
    with open(os.path.join(progpath, 'imdb.pkl'), 'rb') as handle:
        annos = pk.load(handle)

    classdict = annos['classdict']
    classnames = annos['classnames']
    annos_test = annos['annos_test']
    annos_train =annos['annos_train']


