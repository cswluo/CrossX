import scipy.io as sio
import pickle as pk
import os, sys
import os.path as osp

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)


imdbpath = osp.join(progpath, "imdb.pkl")

if not os.path.exists(imdbpath):
    file_list = 'file_list.mat'
    train_list = "train_list.mat"
    test_list = "test_list.mat"

    file_data = sio.loadmat(osp.join(progpath, file_list),
                            mat_dtype=False, squeeze_me=True, matlab_compatible=False,struct_as_record=True)
    train_data = sio.loadmat(osp.join(progpath, train_list),
                            mat_dtype=False, squeeze_me=True, matlab_compatible=False, struct_as_record=True)
    test_data = sio.loadmat(osp.join(progpath, test_list),
                            mat_dtype=False, squeeze_me=True, matlab_compatible=False, struct_as_record=True)

    annos_type=['annotation_list', 'file_list', 'labels']

    train_file_list = train_data['file_list'].tolist()
    train_file_labels = train_data['labels'].tolist()
    test_file_list = test_data['file_list'].tolist()
    test_file_labels = test_data['labels'].tolist()

    class_names = list()
    for file_list in test_file_list:
        prefix = file_list.split('/')[0]
        if prefix not in class_names:
            class_names.append(prefix)

    num_classes = len(class_names)

    class_dict = dict()
    for i in range(num_classes):
        class_dict[i] = class_names[i]



    ####################### save in pickle files
    # classdict: index to class name
    # classnames: class name
    # annos_test: annotations for testing data
    # annos_train: annotations for training data

    with open(os.path.join(progpath, 'imdb.pkl'), 'wb') as handle:
        pk.dump({'classdict': class_dict,
                 'classnames': class_names,
                 'traindata': train_file_list,
                 'trainlabels': train_file_labels,
                 'testdata': test_file_list,
                 'testlabels': test_file_labels}, handle)
else:
    with open(os.path.join(progpath, 'imdb.pkl'), 'rb') as handle:
        annos = pk.load(handle)

    classdict = annos['classdict']
    classnames = annos['classnames']
    traindata = annos['traindata']
    trainlabels =annos['trainlabels']
    testdata = annos['testdata']
    testlabels = annos['testlabels']


