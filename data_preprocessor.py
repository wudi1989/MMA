import numpy as np


def read_rating(path, num_users, num_items, num_total_ratings, a, b, train_ratio):
    # fp = open(path + "ratings-all.txt")

    fp_train = open(path + "train.txt")
    fp_val = open(path + "val.txt")
    fp_test = open(path + "test.txt")

    lines_train = fp_train.readlines()
    lines_val = fp_val.readlines()
    lines_test = fp_test.readlines()

    train_R = np.zeros((num_users, num_items))
    val_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    val_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    num_train_ratings = len(lines_train)
    num_val_ratings = len(lines_val)
    num_test_ratings = len(lines_test)

    ''' Train '''
    for line in lines_train:
        user, item, rating = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_R[user_idx, item_idx] = float(rating.lstrip())
        train_mask_R[user_idx, item_idx] = 1

    '''Validation'''
    for line in lines_val:
        user, item, rating = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        val_R[user_idx, item_idx] = float(rating.lstrip())
        val_mask_R[user_idx, item_idx] = 1

    ''' Test '''
    for line in lines_test:
        user, item, rating = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_R[user_idx, item_idx] = float(rating.lstrip())
        test_mask_R[user_idx, item_idx] = 1

    return train_R, train_mask_R, val_R, val_mask_R, test_R, test_mask_R, num_train_ratings, num_val_ratings, num_test_ratings


def read_decoder(path, num_users, num_items):
    # fp = open(path + "ratings-all.txt")

    fp_train = open(path + "decoder_record.txt")

    lines_test = fp_train.readlines()

    out_mat = np.zeros((num_users, num_items))

    for line in lines_test:
        user, item, rating = line.split("::")
        user_idx = int(user)
        item_idx = int(item)
        out_mat[user_idx, item_idx] = float(rating.lstrip())

    return out_mat
