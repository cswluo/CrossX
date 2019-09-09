######################################
# There are a total of 30 manufacturers, 70 families and 100 variants (categories)
# manufacturer is the superclass of families.
# family is the superclass of variants.
# we mainly do classification on the scale of variant.
#
# the label relationship is:
#           variant < family < manufacturer < car
# the data structure of the traindata, valdata, trainvaldata and testdata:
#       [imgname, variant, family, munufacturer]
######################################


import scipy.io as sio
import pickle as pk
import os, sys
import os.path as osp

progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(progpath)


imdbpath = osp.join(progpath, "imdb.pkl")

if not os.path.exists(imdbpath):

    # variant
    train_variant_list = "annotations/images_variant_train.txt"
    val_variant_list = "annotations/images_variant_val.txt"
    trainval_variant_list = "annotations/images_variant_trainval.txt"
    test_variant_list = "annotations/images_variant_test.txt"

    # family
    train_family_list = "annotations/images_family_train.txt"
    val_family_list = "annotations/images_family_val.txt"
    trainval_family_list = "annotations/images_family_trainval.txt"
    test_family_list = "annotations/images_family_test.txt"

    # manufacturer
    train_manufacturer_list = "annotations/images_manufacturer_train.txt"
    val_manufacturer_list = "annotations/images_manufacturer_val.txt"
    trainval_manufacturer_list = "annotations/images_manufacturer_trainval.txt"
    test_manufacturer_list = "annotations/images_manufacturer_test.txt"

    variants_list = "annotations/variants.txt"
    families_list = "annotations/families.txt"
    manufacturers_list = "annotations/manufacturers.txt"

    # variant, family, manufacturer
    with open(osp.join(progpath, variants_list), 'r') as f:
        classnames_variant = f.readlines()
    with open(osp.join(progpath, families_list), 'r') as f:
        classnames_family = f.readlines()
    with open(osp.join(progpath, manufacturers_list), 'r') as f:
        classnames_manufacturer = f.readlines()
    classnames_variant = [x.rstrip('\n').replace('/', '') for x in classnames_variant]
    classnames_family = [x.rstrip('\n').replace('/', '') for x in classnames_family]
    classnames_manufacturer = [x.rstrip('\n').replace('/', '') for x in classnames_manufacturer]

    # dictionary
    classdict_variant, classdict_family, classdict_manufacturer = dict(), dict(), dict()
    for idx, classname in enumerate(classnames_variant):
        classdict_variant[idx] = classname
    for idx, classname in enumerate(classnames_family):
        classdict_family[idx] = classname
    for idx, classname in enumerate(classnames_manufacturer):
        classdict_manufacturer[idx] = classname

    # for the training set
    traindata = list()
    with open(osp.join(progpath, train_variant_list), 'r') as f:
        rows = f.readlines()
        traindata_variant = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, train_family_list), 'r') as f:
        rows = f.readlines()
        traindata_family = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, train_manufacturer_list), 'r') as f:
        rows = f.readlines()
        traindata_manufacturer = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    if len(traindata_variant) == len(traindata_family) == len(traindata_manufacturer):
        for variant, family, manufacturer in zip(traindata_variant, traindata_family, traindata_manufacturer):
            # print(variant, family, manufacturer)
            variant.append(family[-1])
            variant.append(manufacturer[-1])
            traindata.append(variant)

    # for the validation set
    valdata = list()
    with open(osp.join(progpath, val_variant_list), 'r') as f:
        rows = f.readlines()
        valdata_variant = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, val_family_list), 'r') as f:
        rows = f.readlines()
        valdata_family = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, val_manufacturer_list), 'r') as f:
        rows = f.readlines()
        valdata_manufacturer = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    if len(valdata_variant) == len(valdata_family) == len(valdata_manufacturer):
        for variant, family, manufacturer in zip(valdata_variant, valdata_family, valdata_manufacturer):
            # print(variant, family, manufacturer)
            variant.append(family[-1])
            variant.append(manufacturer[-1])
            valdata.append(variant)


    # for the trainval dataset
    trainvaldata = list()
    with open(osp.join(progpath, trainval_variant_list), 'r') as f:
        rows = f.readlines()
        trainvaldata_variant = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, trainval_family_list), 'r') as f:
        rows = f.readlines()
        trainvaldata_family = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, trainval_manufacturer_list), 'r') as f:
        rows = f.readlines()
        trainvaldata_manufacturer = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    if len(trainvaldata_variant) == len(trainvaldata_family) == len(trainvaldata_manufacturer):
        for variant, family, manufacturer in zip(trainvaldata_variant, trainvaldata_family, trainvaldata_manufacturer):
            # print(variant, family, manufacturer)
            variant.append(family[-1])
            variant.append(manufacturer[-1])
            trainvaldata.append(variant)

    # for the testing dataset
    testdata = list()
    with open(osp.join(progpath, test_variant_list), 'r') as f:
        rows = f.readlines()
        testdata_variant = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, test_family_list), 'r') as f:
        rows = f.readlines()
        testdata_family = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    with open(osp.join(progpath, test_manufacturer_list), 'r') as f:
        rows = f.readlines()
        testdata_manufacturer = [row.rstrip('\n').replace('/', '').split(' ', 1) for row in rows]
    if len(testdata_variant) == len(testdata_family) == len(testdata_manufacturer):
        for variant, family, manufacturer in zip(testdata_variant, testdata_family, testdata_manufacturer):
            # print(variant, family, manufacturer)
            variant.append(family[-1])
            variant.append(manufacturer[-1])
            testdata.append(variant)

    ####################### save in pickle files
    # classdict: index to class name
    # classnames: class name
    # traindata: list of tuples of (imagename, variant)
    # valdata: list of tuples of (imagename, variant)
    # trainvaldata: list of tuples of (imagename, variant)
    # testdata: list of tuples of (imagename, variant)

    with open(os.path.join(progpath, 'imdb.pkl'), 'wb') as handle:
        pk.dump({'classdict_variant': classdict_variant,
                 'classdict_family': classdict_family,
                 'classdict_manufacturer': classdict_manufacturer,
                 'classnames_variant': classnames_variant,
                 'classnames_family': classnames_family,
                 'classnames_manufacturer': classnames_manufacturer,
                 'traindata': traindata,
                 'valdata': valdata,
                 'testdata': testdata,
                 'trainvaldata': trainvaldata}, handle)
else:
    with open(os.path.join(progpath, 'imdb.pkl'), 'rb') as handle:
        annos = pk.load(handle)

    classdict_variant = annos['classdict_variant']
    classdict_family = annos['classdict_family']
    classdict_manufacturer = annos['classdict_manufacturer']
    classnames_variant = annos['classnames_variant']
    classnames_family = annos['classnames_family']
    classnames_manufacturer = annos['classnames_manufacturer']
    traindata = annos['traindata']
    trainvaldata =annos['trainvaldata']
    testdata = annos['testdata']
    valdata = annos['valdata']

print("The vgg aircraft dataset has already been loaded successfully.\n")


