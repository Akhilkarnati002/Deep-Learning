import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks




class BaseModel(ABC):
    """
    This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """


    def __init__(self, opt):


        """        Initialize the BaseModel class."""    

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device=torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.preprocess != 'scale_width': 
            torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0




"""Allows us to save gradients of certain layers for analysis or visualization."""
@staticmethod
def dict_gradient_hook_factory(add_func= lambda x: x):
    saved_dict = {}
    def hook_gen(name):
        def gradient_hook(grad):
            saved_vals = add_func(grad)
            saved_dict[name] = saved_vals

        return gradient_hook
    return hook_gen, saved_dict


@abstractmethod
def set_input(self, input):
    """Unpack input data from the dataloader and perform necessary pre-processing steps.

    Parameters:
        input (dict): include the data itself and its metadata information.

    The option 'direction' can be used to swap images in domain A and domain B.
    """

    pass

@abstractmethod
def forward(self):
    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    pass

@abstractmethod
def optimize_parameters(self):
    """Calculate losses, gradients, and update network weights; called in every training iteration"""
    pass


def setup (self,opt):
    """ Willl load and print networks; create schedulers for task """

    """Parameters:
        opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
    """

    if self.isTrain:
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]


    if not self.isTrain or opt.continue_train:
        load_suffix = opt.epoch
        self.load_networks(load_suffix)



    self.print_networks (opt.verbose)    


def parallelize(self):
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            setattr(self, 'net' + name, networks.DataParallel(net, self.gpu_ids))

def data_dependent_initialize(self, data):
        pass

def evaluate(self):
    """Make models eval mode during test time"""
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            net.eval()


def test(self):
    """Forward function used in test time"""
    with torch.no_grad():
        self.forward()
        self.compute_visuals()


def get_current_visuals(self):
    visual_ret = OrderedDict()
    for name in self.visual_names:
        visual_ret[name] = getattr(self, name)
    return visual_ret

def get_current_losses(self):
    errors_ret = OrderedDict()
    for name in self.loss_names:
        errors_ret[name] = float(getattr(self, 'loss_' + name))
    return errors_ret


def save_networks(self, epoch):
    for name in self.model_names:
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net' + name)
        torch.save(net.cpu().state_dict(), save_path)
        net.cuda(self.gpu_ids[0])

def load_networks(self, epoch):
    for name in self.model_names:
        load_filename = '%s_net_%s.pth' % (epoch, name)
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net' + name)
        state_dict = torch.load(load_path, map_location=str(self.device))
        net.load_state_dict(state_dict)


@staticmethod
def modify_commandline_options(parser, is_train):
    """Add new model-specific options, and rewrite default values for existing options.

    Parameters:
        parser          -- original option parser
        is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

    Returns:
        the modified parser.
    """
    return parser








