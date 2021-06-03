import os
import os.path as osp
import itertools
import pickle
from jax.experimental.optimizers import unpack_optimizer_state, pack_optimizer_state

def save_training_state(ckpt_dir, task, method, opt_state, cur_task_id, cur_epoch, itercount):
    with open(osp.join(ckpt_dir, 'opt_tmp.pickle'), "wb") as fp:
        pickle.dump(unpack_optimizer_state(opt_state), fp)
    with open(osp.join(ckpt_dir, 'epoch_tmp.pickle'), "wb") as fp:
        pickle.dump({'cur_task_id': cur_task_id, 'cur_epoch': cur_epoch, "itercount": itercount}, fp)
    old_task_path, new_task_path = task.save_state(ckpt_dir)
    old_method_path, new_method_path = method.save_state(ckpt_dir)
    os.replace(osp.join(ckpt_dir, 'epoch_tmp.pickle'), osp.join(ckpt_dir, 'epoch.pickle'))
    os.replace(osp.join(ckpt_dir, 'opt_tmp.pickle'), osp.join(ckpt_dir, 'opt.pickle'))
    os.replace(old_task_path, new_task_path)
    os.replace(old_method_path, new_method_path)

def restore_training_state(ckpt_dir, task, method, opt_state, cur_task_id=0, cur_epoch=0, init_seed=0):
    itercount = itertools.count(init_seed)
    restore = False
    if osp.exists(osp.join(ckpt_dir, 'opt.pickle')):
        with open(osp.join(ckpt_dir, 'opt.pickle'), "rb") as fp:
            opt_state = pack_optimizer_state(pickle.load(fp))
        with open(osp.join(ckpt_dir, 'epoch.pickle'), "rb") as fp:
            res = pickle.load(fp)
            cur_task_id, cur_epoch, itercount = res['cur_task_id'], res['cur_epoch'], res["itercount"]
        task.load_state(ckpt_dir)
        method.load_state(ckpt_dir)
        restore = True
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return opt_state, cur_task_id, cur_epoch, itercount, restore