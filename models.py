import tensorflow as tf
import math
import time
import os
import numpy as np


class AutoRec():
    def __init__(self, args, num_users, num_items, train_R, train_mask_R, val_R, val_mask_R, test_R, test_mask_R,
                 num_train_ratings, num_val_ratings, num_test_ratings):
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.train_R = train_R
        self.train_mask_R = train_mask_R
        self.test_R = test_R
        self.test_mask_R = test_mask_R
        self.val_R = val_R
        self.val_mask_R = val_mask_R
        self.num_train_ratings = num_train_ratings
        self.num_test_ratings = num_test_ratings
        self.num_val_ratings = num_val_ratings

        self.hidden_neuron = args.hidden_neuron
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.num_batch = int(math.ceil(self.num_items / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step
        self.random_seed = args.random_seed

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                             self.decay_step, 0.8, staircase=True)
        self.lambda_value = args.lambda_value
        self.grad_clip = args.grad_clip

        self.train_loss_list = []
        self.val_loss_list = []
        self.val_rmse_list = []
        self.val_mae_list = []

        self.test_rmse_list = []
        self.test_mae_list = []

        self.time_list = []
        self.test_prediction = []

    def prepare_model(self, loss_norm, reg_norm):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.train_R, self.train_mask_R, self.val_R, self.val_mask_R, self.test_R, self.test_mask_R))
        train_dataset = dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = dataset.batch(self.batch_size)
        test_dataset = dataset.batch(self.batch_size)
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

        self.train_init_op = iter.make_initializer(train_dataset)
        self.val_init_op = iter.make_initializer(val_dataset)
        self.test_init_op = iter.make_initializer(test_dataset)

        input_R, input_mask_R, input_val_r, input_val_r_mask, input_test_r, input_test_r_mask = iter.get_next()
        input_R, input_mask_R, input_val_r, input_val_r_mask, input_test_r, input_test_r_mask = tf.cast(input_R,
                                                                                                        tf.float32), tf.cast(
            input_mask_R, tf.float32), tf.cast(input_val_r, tf.float32), tf.cast(input_val_r_mask,
                                                                                 tf.float32), tf.cast(input_test_r,
                                                                                                      tf.float32), tf.cast(
            input_test_r_mask, tf.float32)

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(shape=[self.num_users, self.hidden_neuron],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        W = tf.get_variable(name="W", initializer=tf.truncated_normal(shape=[self.hidden_neuron, self.num_users],
                                                                      mean=0, stddev=0.01), dtype=tf.float32)
        mu = tf.get_variable(name="mu", initializer=tf.random_normal(shape=[self.hidden_neuron], stddev=0.01),
                             dtype=tf.float32)
        b = tf.get_variable(name="b", initializer=tf.random_normal(shape=[self.num_users], stddev=0.01),
                            dtype=tf.float32)

        pre_Encoder = tf.matmul(input_R, V) + mu
        self.Encoder = tf.nn.sigmoid(pre_Encoder)
        pre_Decoder = tf.matmul(self.Encoder, W) + b
        self.Decoder = tf.identity(pre_Decoder)

        pre_val_numerator = tf.multiply((self.Decoder - input_val_r), input_val_r_mask)
        self.numerator_val_mae = tf.reduce_sum(tf.abs(pre_val_numerator))
        self.numerator_val_rmse = tf.reduce_sum(tf.square(pre_val_numerator))

        pre_rec_loss = tf.multiply((input_R - self.Decoder), input_mask_R)
        # loss norm
        if loss_norm == 1:
            rec_loss = l1_norm(pre_rec_loss)
        elif loss_norm == 2:
            rec_loss = l2_norm(pre_rec_loss)
        else:
            rec_loss = smooth_l1_norm(pre_rec_loss)
        # reg norm
        if reg_norm == 1:
            pre_reg_loss = l1_norm(W) + l1_norm(V)
        elif reg_norm == 2:
            pre_reg_loss = l2_norm(W) + l2_norm(V)
        else:
            pre_reg_loss = smooth_l1_norm(W) + smooth_l1_norm(V)

        reg_loss = self.lambda_value * pre_reg_loss

        self.loss = rec_loss + reg_loss

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step)

    def run(self, sess, result_path, model_result_path, loss_norm, reg_norm):
        self.sess = sess
        self.result_path = result_path

        self.prepare_model(loss_norm, reg_norm)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        total_time = 0

        count = 0
        min_rmse = 65535
        for epoch_itr in range(self.train_epoch):
            start_time = time.time()
            self.train_model(epoch_itr)
            self.val_model(epoch_itr)
            total_time += (time.time() - start_time)

            self.test_model()
            self.time_list.append(total_time)

            if epoch_itr % 5 == 0 and epoch_itr is not 0:
                if self.val_rmse_list[epoch_itr] < min_rmse:
                    count = 0
                    self.save_model(model_result_path)
                    min_rmse = self.val_rmse_list[epoch_itr]
                else:
                    count += 1
                if count > 4:
                    break
        self.make_records()
        best_rmse_epoch = self.val_rmse_list.index(min(self.val_rmse_list))
        best_test_rmse = self.test_rmse_list[best_rmse_epoch]
        best_mae_epoch = self.val_mae_list.index(min(self.val_mae_list))
        best_test_mae = self.test_mae_list[best_mae_epoch]
        return self.val_rmse_list, str(best_test_rmse), str(best_rmse_epoch), str(best_test_mae), str(
            best_mae_epoch), self.test_prediction, self.time_list

    def train_model(self, itr):
        start_time = time.time()
        self.sess.run(self.train_init_op)
        batch_loss = 0
        for i in range(self.num_batch):
            _, loss = self.sess.run(
                [self.optimizer, self.loss])

            batch_loss = batch_loss + loss
        self.train_loss_list.append(batch_loss)

        if (itr + 1) % self.display_step == 0:
            print("Training //", "Epoch %d //" % (itr), " Total loss = {:.2f}".format(batch_loss),
                  "Elapsed time : %d sec" % (time.time() - start_time))

    def val_model(self, itr):
        start_time = time.time()
        self.sess.run(self.val_init_op)
        numerator_rmse = 0
        numerator_mae = 0
        total_loss = 0
        for i in range(self.num_batch):
            loss, num_rmse, num_mae = self.sess.run([self.loss, self.numerator_val_rmse, self.numerator_val_mae])
            total_loss += loss
            numerator_rmse += num_rmse
            numerator_mae += num_mae
        RMSE = np.sqrt(numerator_rmse / float(self.num_val_ratings))
        MAE = numerator_mae / float(self.num_val_ratings)
        self.val_loss_list.append(total_loss)
        self.val_mae_list.append(MAE)
        self.val_rmse_list.append(RMSE)

        if (itr + 1) % self.display_step == 0:
            # save model

            print("valdating //", "Epoch %d //" % (itr), " Total loss = {:.2f}".format(total_loss),
                  " RMSE = {:.5f}".format(RMSE), " MAE = {:.5f}".format(MAE),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("=" * 50)

    def test_model(self):
        self.sess.run(self.test_init_op)
        total_output = None
        for i in range(self.num_batch):
            loss, output_matrix = self.sess.run(
                [self.loss, self.Decoder])

            if i == 0:
                total_output = output_matrix
            else:
                total_output = np.concatenate((total_output, output_matrix), axis=0)

        self.test_prediction.append(total_output)

        test_numerator = np.multiply((total_output - self.test_R), self.test_mask_R)
        numerator_test_mae = np.sum(np.abs(test_numerator))
        numerator_test_rmse = np.sum(np.square(test_numerator))

        test_RMSE = np.sqrt(numerator_test_rmse / float(self.num_test_ratings))
        test_MAE = numerator_test_mae / float(self.num_test_ratings)
        self.test_rmse_list.append(test_RMSE)
        self.test_mae_list.append(test_MAE)

    def save_model(self, result_path):
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        saver = tf.train.Saver()
        saver.save(self.sess, result_path + "model.ckpt")

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        val_record = self.result_path + "val_record.txt"

        best_val_rmse_time = str(self.time_list[int(self.val_rmse_list.index(min(self.val_rmse_list)))])
        best_val_mae_time = str(self.time_list[int(self.val_mae_list.index(min(self.val_mae_list)))])

        # print("rmse_time:" + rmse_time + ";mae_time:" + mae_time)
        with open(train_record, 'w') as f:
            f.write(str("loss:"))
            f.write('\t')
            for itr in range(len(self.train_loss_list)):
                f.write(str(self.train_loss_list[itr]))
                f.write('\t')
            f.write('\n')

        with open(val_record, 'w') as g:
            g.write(str("loss:"))
            g.write('\t')
            for itr in range(len(self.val_loss_list)):
                g.write(str(self.val_loss_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("RMSE:"))
            for itr in range(len(self.val_rmse_list)):
                g.write(str(self.val_rmse_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("MAE:"))
            for itr in range(len(self.val_mae_list)):
                g.write(str(self.val_mae_list[itr]))
                g.write('\t')
            g.write('\n')

            g.write(str("Best_val_RMSE:"))
            g.write(str(min(self.val_rmse_list)))
            g.write('\n')

            g.write(str("Best_val_RMSE_epoch:"))
            g.write(str(self.val_rmse_list.index(min(self.val_rmse_list))))
            g.write('\n')

            g.write(str("Best_test_RMSE:"))
            g.write(str(self.test_rmse_list[self.val_rmse_list.index(min(self.val_rmse_list))]))
            g.write('\n')

            g.write(str("Best_val_RMSE_time:"))
            g.write(best_val_rmse_time)
            g.write('\n')

            g.write(str("Best_val_MAE:"))
            g.write(str(min(self.val_mae_list)))
            g.write('\n')

            g.write(str("Best_val_MAE_epoch:"))
            g.write(str(self.val_mae_list.index(min(self.val_mae_list))))
            g.write('\n')

            g.write(str("Best_test_MAE:"))
            g.write(str(self.test_mae_list[self.val_mae_list.index(min(self.val_mae_list))]))
            g.write('\n')

            g.write(str("Best_val_MAE_time:"))
            g.write(best_val_mae_time)
            g.write('\n')

        with open(basic_info, 'w') as h:
            h.write(str(self.args))


def l2_norm(tensor):
    return tf.reduce_sum(tf.square(tensor))


def l1_norm(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def smooth_l1_norm(tensor):
    x = tf.abs(tensor)
    x = tf.where(
        tf.less(x, 1),
        tf.square(x),
        x
    )
    return tf.reduce_sum(x)


def save_result(val_rmse_list, val_mae_list, test_rmse_list, test_mae_list, time_list, weight_list, save_path):
    model_list = ["L1_loss L1_reg:", "L1_loss L2_reg:", "L2_loss L1_reg:", "L2_loss L2_reg:"]
    best_val_rmse = str(min(val_rmse_list))
    best_rmse_ep = str(val_rmse_list.index(min(val_rmse_list)))
    best_val_mae = str(min(val_mae_list))
    best_mae_ep = str(val_mae_list.index(min(val_mae_list)))
    best_rmse_time = str(time_list[int(best_rmse_ep)])
    best_mae_time = str(time_list[int(best_mae_ep)])

    best_test_rmse = test_rmse_list[int(best_rmse_ep)]
    best_test_mae = test_mae_list[int(best_mae_ep)]

    # Final result
    print('Epoch:', best_rmse_ep, ' best val rmse:', best_val_rmse, ' best test rmse:', best_test_rmse,
          "best rmse time", best_rmse_time)
    print('Epoch:', best_mae_ep, ' best val mae:', best_val_mae, ' best test mae:', best_test_mae, "best mae time",
          best_mae_time)

    print("|{:.3f}".format(best_test_rmse) + "|{:.3f}".format(best_test_mae) + "|{:.3f}".format(
        float(best_rmse_time)) + "|{:.3f}".format(float(best_mae_time)) + "|")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    val_record = save_path + "val_record.txt"
    weight_record = save_path + "weight_list.txt"

    with open(weight_record, 'w') as g:
        for i in range(4):
            g.write(str(model_list[i]))
            for j in range(len(weight_list)):
                g.write(str(weight_list[j][i]))
                g.write('\t')
            g.write('\n')

    with open(val_record, 'w') as g:
        g.write(str("val rmse list:"))
        g.write('\t')
        for itr in range(len(val_rmse_list)):
            g.write(str(val_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("test rmse list:"))
        g.write('\t')
        for itr in range(len(test_rmse_list)):
            g.write(str(test_rmse_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("val mae list:"))
        g.write('\t')
        for itr in range(len(val_mae_list)):
            g.write(str(val_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("test mae list:"))
        g.write('\t')
        for itr in range(len(test_mae_list)):
            g.write(str(test_mae_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("time list:"))
        g.write('\t')
        for itr in range(len(time_list)):
            g.write(str(time_list[itr]))
            g.write('\t')
        g.write('\n')

        g.write(str("Best_val_RMSE:"))
        g.write(best_val_rmse)
        g.write('\n')

        g.write(str("Best_RMSE_epoch:"))
        g.write(best_rmse_ep)
        g.write('\n')

        g.write(str("Best_test_RMSE:"))
        g.write(str(best_test_rmse))
        g.write('\n')

        g.write(str("Best_RMSE_time:"))
        g.write(best_rmse_time)
        g.write('\n')

        g.write(str("Best_val_MAE:"))
        g.write(best_val_mae)
        g.write('\n')

        g.write(str("Best_MAE_epoch:"))
        g.write(best_mae_ep)
        g.write('\n')

        g.write(str("Best_test_MAE:"))
        g.write(str(best_test_mae))
        g.write('\n')

        g.write(str("Best_MAE_time:"))
        g.write(best_mae_time)
        g.write('\n')
