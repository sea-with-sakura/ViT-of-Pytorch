import os
import json
import pandas as pd
import torch
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import importlib

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    # add datetime postfix
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    exp_name = config.exp_name + '_{}_bs{}_lr{}_wd{}'.format(config.dataset, config.batch_size, config.lr, config.wd)
    exp_name += ('_' + timestamp)

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join('experiments', 'tb', exp_name)
    config.checkpoint_dir = os.path.join('experiments', 'save', exp_name, 'checkpoints/')
    config.result_dir = os.path.join('experiments', 'save', exp_name, 'results/')
    for dir in [config.summary_dir, config.checkpoint_dir, config.result_dir]:
        ensure_dir(dir)

    # save config
    write_json(vars(config), os.path.join('experiments', 'save', exp_name, 'config.json'))

    return config


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data.loc[:, col] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class TensorboardWriter():
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""

        if enabled:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                print(message)

        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class SwanLabWriter():
    def __init__(self, log_dir, enabled):
        self.enabled = enabled
        self.step = 0
        self.mode = ''
        self.timer = datetime.now()
        
        if enabled:
            try:
                import swanlab
                self.swanlab = swanlab
                # Initialize SwanLab with project name
                self.swanlab.init(project="vision-transformer", config={"log_dir": log_dir})
                print("SwanLab initialized successfully")
            except ImportError:
                print("Warning: SwanLab is not installed. Please install it with 'pip install swanlab'")
                self.enabled = False
                self.swanlab = None

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.log_metric('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def add_scalar(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            self.swanlab.log({tag: data}, step=self.step)

    def log_metric(self, tag, data, *args, **kwargs):
        """Alias for add_scalar to maintain compatibility"""
        self.add_scalar(tag, data, *args, **kwargs)

    def add_scalars(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                for key, value in data.items():
                    new_tag = '{}/{}/{}'.format(tag, key, self.mode)
                    self.swanlab.log({new_tag: value}, step=self.step)
            else:
                self.swanlab.log(data, step=self.step)

    def add_image(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            self.swanlab.log({tag: self.swanlab.Image(data)}, step=self.step)

    def add_images(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            self.swanlab.log({tag: [self.swanlab.Image(img) for img in data]}, step=self.step)

    def add_text(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            self.swanlab.log({tag: data}, step=self.step)

    def add_histogram(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # SwanLab doesn't have direct histogram support, but we can log as text
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            try:
                # Convert histogram data to text summary
                hist_data = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                self.swanlab.log({tag: hist_data}, step=self.step)
            except:
                pass  # Skip if data can't be processed

    def add_pr_curve(self, tag, data, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # Add mode(train/valid) tag
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            self.swanlab.log({tag: data}, step=self.step)

    def add_embedding(self, tag, mat, *args, **kwargs):
        if self.enabled and self.swanlab is not None:
            # SwanLab doesn't have direct embedding support
            # We'll log the shape information instead
            if self.mode:
                tag = '{}/{}'.format(tag, self.mode)
            try:
                self.swanlab.log({tag: {"shape": list(mat.shape)}}, step=self.step)
            except:
                pass  # Skip if data can't be processed

    def finish(self):
        if self.enabled and self.swanlab is not None:
            self.swanlab.finish()

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return corresponding SwanLab method or a no-op
        Otherwise:
            return a blank function handle that does nothing
        """
        if self.enabled and self.swanlab is not None:
            # Return a wrapper that logs to SwanLab
            def wrapper(*args, **kwargs):
                try:
                    method = getattr(self.swanlab, name, None)
                    if method and callable(method):
                        return method(*args, **kwargs)
                except:
                    pass
            return wrapper
        else:
            # Return a no-op function
            def no_op(*args, **kwargs):
                pass
            return no_op


def log_model_layers(model, config):
    """Log model layers to JSON file"""
    layer_names = []
    for name, _ in model.named_modules():
        layer_names.append(name)
    json_path = os.path.join(config.checkpoint_dir, 'model_layers.json')
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(layer_names, f, ensure_ascii=False, indent=4)
    print(f"模型各层名称已保存至 {json_path}")


def load_checkpoint(path):
    """ Load weights from a given checkpoint path in npz/pth """
    if path.endswith('pth'):
        state_dict = torch.load(path)['state_dict']
    else:
        raise ValueError("checkpoint format {} not supported yet!".format(path.split('.')[-1]))

    return state_dict