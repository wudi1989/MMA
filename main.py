from decimal import Decimal
import tensorflow as tf
from data_preprocessor import *
from models import AutoRec, save_result
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
import argparse


current_time = time.time()
parser = argparse.ArgumentParser(description='MMA')
parser.add_argument('--data_name', type=str, choices= ["Ml1M", "Ml100k", "Yahoo", "Hetrec-ML"], default="Ml1M")
parser.add_argument('--hidden_neuron', type=int, default=500)
parser.add_argument('--lambda_value', type=float, default=0.001)

parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=700)

parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
parser.add_argument('--grad_clip', type=bool, default=False)
parser.add_argument('--base_lr', type=float, default=1e-3)
parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--display_step', type=int, default=1)

args = parser.parse_args()
tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

data_name = args.data_name

if data_name == "Ml1M":
    num_users = 6040
    num_items = 3952
    num_total_ratings = 1000209
    norm_list = [[1, 1], [1, 2], [2, 1], [2, 2]]
    lambda_list = [0.01, 0.001, 30, 30]

elif data_name == "Ml100k":
    num_users = 943
    num_items = 1682
    num_total_ratings = 100000
    norm_list = [[1, 1], [1, 2], [2, 1], [2, 2]]
    lambda_list = [0.1, 0.001,10,20]
elif data_name == "Yahoo":
    num_users = 15400
    num_items = 1000
    num_total_ratings = 365704
    norm_list = [[1, 1], [1, 2], [2, 1], [2, 2]]
    lambda_list = [0.1, 10, 1, 15]
else:
    num_users = 2113
    num_items = 10109
    num_total_ratings = 855598
    norm_list = [[1, 1], [1, 2], [2, 1], [2, 2]]
    lambda_list = [1e-6, 0.1, 1e-5, 0.1]

train_ratio = 0.9
path = "data_7_1_2/%s" % data_name + "/"


train_R, train_mask_R, val_R, val_mask_R, test_R, test_mask_R, num_train_ratings, num_val_ratings, num_test_ratings = read_rating(
    path, num_users, num_items, num_total_ratings, 1, 0, train_ratio)

val_result_list = []
best_mae_result_list = []
best_rmse_result_list = []
total_test_output = []
time_list = []

cumulative_time = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for i in range(4):
    loss_norm = norm_list[i][0]
    reg_norm = norm_list[i][1]
    args.lambda_value = lambda_list[i]
    result_path = 'results/' + data_name + '/' + str(args.optimizer_method) + '_lr_' + str(
        args.base_lr) + "AutoRec_l" + str(loss_norm) + "_l" + str(reg_norm) + "/"
    model_result_path = 'results/models/' + data_name + '/' + str(args.optimizer_method) + '_lr_' + str(
        args.base_lr) + "AutoRec_l" + str(loss_norm) + "_l" + str(reg_norm) + "/"
    with tf.Session(config=config) as sess:
        AutoRec1 = AutoRec(args, num_users, num_items, train_R.T, train_mask_R.T, val_R.T, val_mask_R.T,
                           test_R.T,
                           test_mask_R.T,
                           num_train_ratings, num_val_ratings, num_test_ratings)
        # run
        val_result, test_RMSE_result, epoch_RMSE_index, test_MAE_result, epoch_MAE_index, test_output, running_time_list = AutoRec1.run(
            sess,
            result_path,
            model_result_path,
            loss_norm,
            reg_norm,
        )

        if running_time_list[int(epoch_RMSE_index)] > cumulative_time:
            cumulative_time = running_time_list[int(epoch_RMSE_index)]

        best_val_result = val_result[:int(epoch_RMSE_index) + 1]
        final_test_output = test_output[:int(epoch_RMSE_index) + 1]
        val_result_list.append(best_val_result)
        total_test_output.append(final_test_output)

        best_mae_result_list.append(str(Decimal(test_MAE_result).quantize(Decimal('0.000'))))
        best_rmse_result_list.append(str(Decimal(test_RMSE_result).quantize(Decimal('0.000'))))

        sess.close()
    tf.reset_default_graph()

print("test_RMSE:" + str(best_rmse_result_list).replace("'", "|").replace(",", "").replace("| |", "|"))
print("test_MAE:" + str(best_mae_result_list).replace("'", "|").replace(",", "").replace("| |", "|"))

# constant
zeta = np.sqrt(1 / np.log(args.train_epoch))

total_Cl_list = []
total_alpha_list = []
total_weight_list = []
test_rmse_list = []
test_mae_list = []

val_rmse_list = []
val_mae_list = []

for j in range(args.train_epoch):
    # ensemble weight
    alpha_list = []
    weight_list = []

    temp_CL_list = []
    start_time = time.time()

    for i in range(4):
        if j > 0:
            if j < len(val_result_list[i]):
                temp_CL_list.append(total_Cl_list[j - 1][i] + val_result_list[i][j])
            else:
                temp_CL_list.append(total_Cl_list[j - 1][i] + val_result_list[i][-1])
        else:
            temp_CL_list.append(np.sum(val_result_list[i][:j + 1]))
    total_Cl_list.append(temp_CL_list)
    pre_numerator = []
    for i in range(4):
        pre_numerator.append(np.exp(-zeta * total_Cl_list[j][i]))
    numerator = np.sum(pre_numerator)
    for i in range(4):
        weight_list.append(pre_numerator[i] / numerator)

    final_result = 0

    for i in range(0, 4):
        if j < len(val_result_list[i]):
            final_result += np.multiply(weight_list[i], total_test_output[i][j])
        else:
            final_result += np.multiply(weight_list[i], total_test_output[i][-1])

    val_numerator = np.multiply((val_R.T - final_result), val_mask_R.T)
    val_mae = np.sum(np.abs(val_numerator)) / float(num_val_ratings)
    val_rmse = np.sqrt(np.sum(np.square(val_numerator)) / float(num_val_ratings))

    total_time = time.time() - start_time
    cumulative_time += total_time

    time_list.append(cumulative_time)

    test_numerator = np.multiply((test_R.T - final_result), test_mask_R.T)
    test_mae = np.sum(np.abs(test_numerator)) / float(num_test_ratings)
    test_rmse = np.sqrt(np.sum(np.square(test_numerator)) / float(num_test_ratings))

    total_alpha_list.append(alpha_list)
    total_weight_list.append(weight_list)

    val_rmse_list.append(val_rmse)
    val_mae_list.append(val_mae)

    test_rmse_list.append(test_rmse)
    test_mae_list.append(test_mae)

result_path = 'results/' + data_name + '/' + str(args.optimizer_method) + '_lr_' + str(
    args.base_lr) + "_ensemble_result/"
save_result(val_rmse_list, val_mae_list, test_rmse_list, test_mae_list, time_list, total_weight_list, result_path)
