import numpy as np
import tensorflow as tf
from pyemd import emd

def masked_mse_tf(preds, labels, mask=None):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    shape = preds.get_shape()
    if len(shape) > 4:
        batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.get_shape()
        preds = tf.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
        labels = tf.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
    if mask is None:
        mask = ~tf.is_nan(labels)
    mask = tf.cast(mask, tf.float32)
    # mask /= tf.reduce_sum(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_mae_tf(preds, labels, mask=None):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """

    shape = preds.get_shape()
    if len(shape) > 3:
        batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.get_shape()
        preds = tf.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
        labels = tf.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
    if mask is None:
        mask = ~tf.is_nan(labels)

    mask = tf.cast(mask, tf.float32)
    # mask /= tf.reduce_sum(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_mape_tf(preds, labels, mask=None):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """

    shape = preds.get_shape()
    if len(shape) > 3:
        batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.get_shape()
        preds = tf.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
        labels = tf.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
    if mask is None:
        mask = ~tf.is_nan(labels)

    mask = tf.cast(mask, tf.float32)
    # mask /= tf.reduce_sum(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels)) / preds
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_rmse_tf(preds, labels, mask=None):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, mask=mask))

# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels, mask):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_tf(preds=preds, labels=labels, mask=mask)

    return loss

def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels, mask):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_tf(preds=preds, labels=labels, mask=mask)

    return loss

def masked_mae_loss(scaler, null_val):
    def loss(preds, labels, mask):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, mask=mask)

        return mae

    return loss

def masked_mape_loss(scaler, null_val):
    def loss(preds, labels, mask):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mape_tf(preds=preds, labels=labels, mask=mask)
        return mae

    return loss

def masked_kl_loss(preds, labels, mask, epsilon=1e-3):
    """
    calculate the kl loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    log_pred = tf.log(preds + epsilon)
    log_true = tf.log(labels + epsilon)
    log_sub = tf.subtract(log_pred, log_true)
    mul_op = tf.multiply(preds, log_sub)
    sum_hist = tf.reduce_sum(mul_op, -1)
    mask = tf.cast(mask, tf.float32)
    sum_hist = tf.multiply(mask, sum_hist)
    sum_hist = tf.where(tf.is_nan(sum_hist), tf.zeros_like(sum_hist), sum_hist)
    weight_avg_kl_div = tf.reduce_sum(sum_hist)
    avg_kl_div = weight_avg_kl_div / tf.reduce_sum(mask)

    return avg_kl_div

def masked_l2_loss(preds, labels, mask):
    """
    calculate the l2 loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    sub_square = tf.square(preds - labels)
    sum_hist = tf.reduce_sum(sub_square, -1)
    mask = tf.cast(mask, tf.float32)
    sum_hist = tf.multiply(mask, sum_hist)

    weight_avg_l2_div = tf.reduce_sum(sum_hist)
    avg_l2_div = weight_avg_l2_div / tf.reduce_sum(mask)

    return avg_l2_div

def masked_emd_loss(preds, labels, mask):
    """
    calculate the emd loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """
    B = int(preds.get_shape()[-1])

    d_matrix = np.ones((B, B), np.float32)
    for i in range(B):
        d_matrix[i+1:, i] = 0
    d_tensor = tf.convert_to_tensor(d_matrix)
    diff = labels - preds
    diff = tf.reshape(diff, [-1, B])
    diff = tf.matmul(diff, d_tensor)
    ypred = tf.reduce_sum(diff, axis=-1)
    mask = tf.cast(mask, tf.float32)
    weight = tf.reshape(mask, [-1, 1])
    wasser = weight * ypred
    avg_wasser = tf.reduce_sum(wasser) / tf.reduce_sum(weight)

    return avg_wasser

def masked_fml2_loss(preds, labels, mask, o_output, d_output):
    """
    calculate the l2 loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    sub_square = tf.square(preds - labels)
    sum_hist = tf.reduce_sum(sub_square, -1)
    mask = tf.cast(mask, tf.float32)
    sum_hist = tf.multiply(mask, sum_hist)

    weight_avg_l2_div = tf.reduce_sum(sum_hist)
    avg_l2_div = weight_avg_l2_div / tf.reduce_sum(mask)


    return avg_l2_div

def masked_emd_loss_error(preds, labels, mask):
    """
    calculate the emd loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """
    B = int(preds.get_shape()[-1])
    ypred = tf.reshape(tf.square(labels - preds), [-1, B])
    mask = tf.cast(mask, tf.float32)
    weight = tf.reshape(mask, [-1, 1])
    x_values = tf.range(B)
    x_values = tf.reshape(x_values, [-1, 1])
    x_values = tf.cast(x_values, tf.float32)
    weighted_x = tf.matmul(ypred, x_values)
    wasser = weight * weighted_x
    avg_wasser = tf.reduce_sum(wasser) / tf.reduce_sum(weight)

    return avg_wasser

# calculate for evaluation dataset
def calculate_metrics(df_pred, df_test, mask):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), mask=mask)
    mae = masked_mae_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), mask=mask)
    rmse = masked_rmse_np(preds=df_pred.as_matrix(), labels=df_test.as_matrix(), mask=mask)
    return mae, mape, rmse


def masked_rmse_np(preds, labels, mask=None):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, mask=mask))

def masked_mse_np(preds, labels, mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = preds.get_shape()
        if len(shape) > 4:
            batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.shape
            preds = np.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
            labels = np.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])

        if mask is None:
            mask = ~np.is_nan(labels)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = preds.get_shape()
        if len(shape) > 4:
            batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.shape
            preds = np.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
            labels = np.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
        if mask is None:
            mask = ~np.is_nan(labels)

        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, mask=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        shape = preds.get_shape()
        if len(shape) > 4:
            batch_size, horizon, nb_nodes, nb_nodes, o_dim = preds.shape
            preds = np.reshape(preds, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
            labels = np.reshape(labels, [int(batch_size), int(horizon), int(nb_nodes), int(nb_nodes)])
        if mask is None:
            mask = ~np.is_nan(labels)

        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def calculate_metrics_hist(pred, label, weight):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    kl = masked_kl_np(preds=pred, labels=label, mask=weight)
    # label_kl = masked_kl_np(preds=label, labels=pred, mask=weight)
    l2 = masked_l2_np(preds=pred, labels=label, mask=weight)
    emd = masked_emd_np2(preds=pred, labels=label, mask=weight)
    jsd = masked_jsd_np(preds=pred, labels=label, mask=weight)

    return kl, l2, emd, jsd

def masked_kl_np(preds, labels, mask, epsilon=1e-3):
    """
    calculate the kl loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    log_pred = np.log(preds + epsilon)
    log_true = np.log(labels + epsilon)
    log_sub = log_pred - log_true
    mul_op = np.multiply(preds, log_sub)
    sum_hist = np.sum(mul_op, -1)
    mask = mask.astype(int)
    # print("Sum HIst shape is ", sum_hist.shape)
    # print("mask shape is ", mask.shape)

    sum_hist = mask * sum_hist
    weight_avg_kl_div = np.sum(sum_hist)
    avg_kl_div = weight_avg_kl_div / np.sum(mask)

    return avg_kl_div

def masked_jsd_np(preds, labels, mask, epsilon=1e-3):
    """
    calculate the kl loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    mean_hist = (preds + labels) / 2
    log_mean = np.log(mean_hist + epsilon)
    log_pred = np.log(preds + epsilon)
    log_true = np.log(labels + epsilon)

    log_pred_mean = log_pred - log_mean
    log_label_mean = log_true - log_mean

    mul_pred_mean = np.multiply(preds, log_pred_mean)
    mul_label_mean = np.multiply(labels, log_label_mean)

    sum_1 = np.sum(mul_pred_mean, axis=-1)
    sum_2 = np.sum(mul_label_mean, axis=-1)
    jsd_sum = 0.5 * sum_1 + 0.5*sum_2

    mask = mask.astype(int)

    sum_hist = mask * jsd_sum
    weight_avg_js_div = np.sum(sum_hist)
    avg_js_div = weight_avg_js_div / np.sum(mask)

    return avg_js_div

def masked_l2_np(preds, labels, mask):
    """
    calculate the l2 loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    sub_square = np.square(preds - labels)
    sum_hist = np.sum(sub_square, -1)
    mask = mask.astype(np.int8)
    sum_hist = np.multiply(mask, sum_hist)

    weight_avg_l2_div = np.sum(sum_hist)
    avg_l2_div = weight_avg_l2_div / np.sum(mask)

    return avg_l2_div

def masked_emd_np(preds, labels, mask):
    """
    calculate the emd loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """
    B = preds.shape[-1]

    ypred = np.reshape(np.square(labels - preds), [-1, B])
    weight = np.reshape(mask, [-1, 1])
    x_values = np.arange(B)
    x_values = np.reshape(x_values, [-1, 1])
    x_values = x_values.astype(np.float32)
    weighted_x = np.matmul(ypred, x_values)
    wasser = weight * weighted_x
    avg_wasser = np.sum(wasser) / np.sum(weight)

    return avg_wasser

def masked_emd_np2(preds, labels, mask):
    """
    calculate the emd loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """
    B = preds.shape[-1]
    preds = preds[mask, :]
    labels = labels[mask, :]
    preds = np.reshape(preds, [-1, B])
    labels = np.reshape(labels, [-1, B])

    return weighted_emd(labels, preds).mean()

def weighted_emd(y_true, y_pred):
    n_bucket = y_true.shape[1]
    nb_result = y_true.shape[0]
    d_matrix = np.zeros((n_bucket, n_bucket))
    for i in range(n_bucket):
        d_matrix[i, i:n_bucket] = np.arange(n_bucket)[:n_bucket - i]
    d_matrix = np.maximum(d_matrix, d_matrix.T)
    emds = []
    for j in range(nb_result):
        hist_true = y_true[j, :].astype(np.float64)
        hist_pred = y_pred[j, :].astype(np.float64)
        emd_j = emd(hist_pred, hist_true, d_matrix)
        emds.append(emd_j)
    emds = np.array(emds)

    return emds

def calculate_metrics_hist_matrix(pred, label):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    kl = masked_kl_np_matrix(preds=pred, labels=label)
    jsd = masked_jsd_np_matrix(preds=label, labels=pred)
    emd = masked_emd_np_matrix(preds=pred, labels=label)

    return kl, jsd, emd

def masked_kl_np_matrix(preds, labels, epsilon=1e-3):
    """
    calculate the kl loss

    :param preds:
    :param labels:
    :return:
    """

    log_pred = np.log(preds + epsilon)
    log_true = np.log(labels + epsilon)
    log_sub = log_pred - log_true
    mul_op = np.multiply(preds, log_sub)
    sum_hist = np.sum(mul_op, -1)

    return sum_hist

def masked_jsd_np_matrix(preds, labels, epsilon=1e-3):
    """
    calculate the kl loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    mean = 0.5 * (preds + labels)
    kl_pred = masked_kl_np_matrix(preds, mean, epsilon)
    kl_label = masked_kl_np_matrix(labels, mean, epsilon)

    return (kl_pred + kl_label) / 2

def masked_emd_np_matrix(preds, labels):
    """
    calculate the emd loss

    :param preds:
    :param labels:
    :param mask:
    :return:
    """

    preds_shape = preds.shape
    emd_matrix = np.zeros(preds_shape[:-1], dtype=np.float32)
    for i in range(preds_shape[1]):
        for j in range(preds_shape[2]):
            emd_matrix[:, i, j] = weighted_emd(preds[:, i, j, :], labels[:, i, j, :])

    return emd_matrix