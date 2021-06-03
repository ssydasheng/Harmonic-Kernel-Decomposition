import os.path as osp
import shutil
import time
import numpy as np
import tensorflow as tf

import sklearn
from scipy.stats import norm
from scipy.misc import logsumexp


def train_epoch(args, print_func, print_dict, train_op, summary_op, summary_writer, global_step, sess, epoch, N, print_every=1):
    time_epoch = time.time()
    names, values = zip(*print_dict.items())
    names, values = list(names), list(values)
    averages = [[] for _ in range(len(names))]
    if args.batch_size is None: n_iter = 1
    else: n_iter = int(np.ceil(N / float(args.batch_size)))
    # run_metadata = tf.RunMetadata()

    for t in range(n_iter):
        if isinstance(train_op, (list, tuple)):
            res = sess.run([train_op[0], summary_op, global_step] + values) # train_q_op
            sess.run(train_op[1]) # train_m_op
        else:
            res = sess.run([train_op, summary_op, global_step] + values)

        if summary_writer is not None and res[2] % 100 == 0:
            summary_writer.add_summary(res[1], global_step=res[2])
        for a, r in zip(averages, res[3:]):
            a.append(r)
        # for n, r in zip(names, res[3:]):
        #     print_func('Iter {}: {} = {:.5f}'.format(t, n, r))

        # if t == 50:
        #     sess.run([summary_op] + values,
        #              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #              run_metadata=run_metadata)
        #     from tensorflow.python.client import timeline
        #     trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #     trace_file = open('no-bp.ctf.json', 'w')
        #     trace_file.write(trace.generate_chrome_trace_format())
        #     exit(0)

    averages = [np.mean(a) for a in averages]
    format_ = ' | '.join(['{}={:.5f}'.format(n, l) for n, l in zip(names, averages)])
    elapsed_time = (time.time() - time_epoch) / n_iter
    if epoch % print_every == 0:
        print_func('>>> Train --  Epoch {:5d}/{:5d} -- {:5f}s/iter | '.format(epoch, args.epochs, elapsed_time) + format_)
    return dict(zip(names, averages)), elapsed_time


def setup_train_summary(model, summary_path, summary_dict, summary=True):
    if not summary:
        return tf.no_op(), None
    for name, value in summary_dict.items():
        if len(value.shape) == 0:
           tf.summary.scalar(name, value, family='train')
        else:
            tf.summary.histogram(name, value, family='train')
    summary_op = tf.summary.merge_all()
    if osp.exists(summary_path):
        shutil.rmtree(summary_path)
    summary_writer = tf.summary.FileWriter(summary_path)
    return summary_op, summary_writer

def write_test_summary(test_writer, names, averages, epoch, prefix='test'):
    if test_writer is None:
        return
    summary = tf.Summary()
    for name, average in zip(names, averages):
        summary.value.add(tag=prefix+"/"+name, simple_value=average)
    test_writer.add_summary(summary, epoch)

def test_epoch(args, print_func, x_pred, y_pred, test_dict, summary_writer, sess, epoch, global_step,
               test_x, test_y, PREFIX='TEST DATA', return_all_averages=False, print_res=True):
    time_epoch = time.time()
    names, values = zip(*test_dict.items())
    names, values = list(names), list(values)
    all_averages = [[] for _ in range(len(names))]

    n_batches = max(int(np.ceil(test_x.shape[0] / args.test_batch_size)), 1)
    for iter in range(n_batches):
        batch_x = test_x[iter * args.test_batch_size: (iter + 1) * args.test_batch_size]
        if y_pred is not None:
            batch_y = test_y[iter * args.test_batch_size: (iter + 1) * args.test_batch_size]
            res = sess.run(values, feed_dict={x_pred: batch_x, y_pred: batch_y})
        else:
            res = sess.run(values, feed_dict={x_pred: batch_x})
        for a, r in zip(all_averages, res):
            a.append(r)

    averages = [np.mean(np.concatenate(a, 0)) for a in all_averages]
    format_ = ' | '.join(['{}={:.5f}'.format(n, l) for n, l in zip(names, averages)])
    elapsed_time = time.time() - time_epoch
    if print_res:
        print_func('>>> TEST [{}] Epoch {:5d}/{:5d} -- {:5f}s | '.format(PREFIX, epoch, args.epochs, elapsed_time) + format_)
    write_test_summary(summary_writer, names, averages, sess.run(global_step), prefix=PREFIX)
    if not return_all_averages:
        return dict(zip(names, averages))
    all_averages = [np.concatenate(a, 0) for a in all_averages]
    return dict(zip(names, averages)), dict(zip(names, all_averages))