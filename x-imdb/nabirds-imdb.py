import scipy.io as sio
import pickle as pk
import os, sys
import os.path as osp

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)


imdbpath = osp.join(progpath, "imdb.pkl")

if not os.path.exists(imdbpath):

    img_list = "images.txt"
    tvt_list = "train_test_split.txt"
    cls_list = "classes.txt"
    img_cls_list = "image_class_labels.txt"
    cls_hir_list = "hierarchy.txt"


    classnames = {}
    with open(os.path.join(progpath, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            classnames[class_id] = ' '.join(pieces[1:])


    classparents = {}
    with open(os.path.join(progpath, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            classparents[child_id] = parent_id


    imgpaths = {}
    with open(os.path.join(progpath, img_list)) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join('images', pieces[1])
            imgpaths[image_id] = path


    imglabels = {}
    with open(os.path.join(progpath, 'image_class_labels.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = pieces[1]
            imglabels[image_id] = class_id

    traindata, testdata = list(), list()
    with open(os.path.join(progpath, 'train_test_split.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            is_train = int(pieces[1])

            clsid = imglabels[image_id]
            clsname = classnames[clsid]
            prntid = classparents[clsid]
            prntname = classnames[prntid]

            if is_train:
                traindata.append([imgpaths[image_id], clsid, clsname, prntid, prntname])
            else:
                testdata.append([imgpaths[image_id], clsid, clsname, prntid, prntname])


    num_classes = len(classnames)
    num_subclasses = len(set(imglabels.values()))
    num_prntclasses = len(set(classparents.values()))
    print(num_classes, num_subclasses, num_prntclasses)
    ####################### save in pickle files
    # prntclassid: classparents[sub_class_id] = parent_class_id
    # classnames: classnames[sub_class_id/parent_class_id] = class_name
    # subclassid: imglabels[img_id] = sub_class_id
    # traindata: [imgpath, subclass_id, subclass_name, parent_class_id, parent_class_name]
    # testdata: [imgpath, subclass_id, subclass_name, parent_class_id, parent_class_name]

    with open(os.path.join(progpath, 'imdb.pkl'), 'wb') as handle:
        pk.dump({'prntclassid': classparents,
                 'classnames': classnames,
                 'subclassid': imglabels,
                 'testdata': testdata,
                 'traindata': traindata}, handle)
else:
    with open(os.path.join(progpath, 'imdb.pkl'), 'rb') as handle:
        annos = pk.load(handle)

    prntclassid = annos['prntclassid']
    classnames = annos['classnames']
    subclassid = annos['subclassid']
    testdata = annos['testdata']
    traindata = annos['traindata']


