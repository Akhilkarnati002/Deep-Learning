import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import Network  # Ensure Network.py is imported correctly


class BaseModel(ABC):
    """
    Abstract Base Class for models.
    Subclasses must implement:
        - set_input
        - forward
        - optimize_parameters
        - compute_visuals
        - modify_commandline_options (optional)
    """

    def __init__(self, opt):
        """Initialize the BaseModel class."""

        self.opt = opt
        self.isTrain = getattr(opt, 'isTrain', True)
        self.gpu_ids = getattr(opt, 'gpu_ids', [])
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.gpu_ids else 'cpu')

        # Directory for saving checkpoints
        checkpoints_dir = getattr(opt, 'checkpoints_dir', 'checkpoints')
        name = getattr(opt, 'name', 'default_model')
        self.save_dir = os.path.join(checkpoints_dir, name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Lists for tracking model state
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    def move_networks_to_device(self):
        """Move all networks listed in self.model_names to self.device."""
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            if torch.cuda.is_available() and self.gpu_ids:
                net = torch.nn.DataParallel(net, device_ids=self.gpu_ids)
            net.to(self.device)
            setattr(self, 'net' + name, net)

    @abstractmethod
    def set_input(self, input):
        """Unpack input data and preprocess. Must be implemented by subclass."""
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass. Must be implemented by subclass."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Compute losses, gradients, and update network weights."""
        pass

    @abstractmethod
    def compute_visuals(self):
        """Calculate visuals for display or logging."""
        pass

    def test(self):
        """Forward function used during test/evaluation."""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def get_current_visuals(self):
        """Return current visuals as an OrderedDict."""
        return OrderedDict((name, getattr(self, name)) for name in self.visual_names)

    def get_current_losses(self):
        """Return current losses as an OrderedDict."""
        errors = OrderedDict()
        for name in self.loss_names:
            loss_attr = 'loss_' + name
            errors[name] = float(getattr(self, loss_attr, 0.0))
        return errors

    def save_networks(self, epoch_label):
        """Save all networks to disk."""
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            save_filename = f'{epoch_label}_net_{name}.pth'
            save_path = os.path.join(self.save_dir, save_filename)

            # Handle DataParallel
            if isinstance(net, torch.nn.DataParallel):
                net_to_save = net.module
            else:
                net_to_save = net

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net_to_save.cpu().state_dict(), save_path)
            net_to_save.to(self.device)

    def load_networks(self, epoch_label):
        """Load all networks from disk."""
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            load_filename = f'{epoch_label}_net_{name}.pth'
            load_path = os.path.join(self.save_dir, load_filename)
            if not os.path.exists(load_path):
                print(f'[Warning] {load_path} does not exist. Skipping loading.')
                continue

            # Handle DataParallel
            if isinstance(net, torch.nn.DataParallel):
                net_to_load = net.module
            else:
                net_to_load = net

            print(f'Loading network {name} from {load_path} to {self.device}')
            state_dict = torch.load(load_path, map_location=self.device)
            net_to_load.load_state_dict(state_dict)
            net_to_load.to(self.device)

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Add model-specific options and set defaults."""
        return parser
