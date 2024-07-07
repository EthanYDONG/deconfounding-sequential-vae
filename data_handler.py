import numpy as np
import torch

class DataHandler(object):
    def __init__(self, opts):
        self.train_num = None
        self.validation_num = None
        self.test_num = None
        self.x_train = None
        self.a_train = None
        self.r_train = None
        self.mask_train = None
        self.x_validation = None
        self.a_validation = None
        self.r_validation = None
        self.mask_validation = None
        self.x_test = None
        self.a_test = None
        self.r_test = None
        self.mask_test = None
        self.train_r_max = None
        self.train_r_min = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_dataset(opts)

    def load_dataset(self, opts):
        data_train = np.load(opts['training_data'])
        self.x_train = torch.tensor(data_train['x_train'], dtype=torch.float32).to(self.device)
        self.a_train = torch.tensor(data_train['a_train'], dtype=torch.float32).to(self.device)
        self.r_train = torch.tensor(data_train['r_train'], dtype=torch.float32).to(self.device)
        self.mask_train = torch.tensor(data_train['mask_train'], dtype=torch.float32).to(self.device)

        data_validation = np.load(opts['validation_data'])
        self.x_validation = torch.tensor(data_validation['x_validation'], dtype=torch.float32).to(self.device)
        self.a_validation = torch.tensor(data_validation['a_validation'], dtype=torch.float32).to(self.device)
        self.r_validation = torch.tensor(data_validation['r_validation'], dtype=torch.float32).to(self.device)
        self.mask_validation = torch.tensor(data_validation['mask_validation'], dtype=torch.float32).to(self.device)

        data_test = np.load(opts['testing_data'])
        self.x_test = torch.tensor(data_test['x_test'], dtype=torch.float32).to(self.device)
        self.a_test = torch.tensor(data_test['a_test'], dtype=torch.float32).to(self.device)
        self.r_test = torch.tensor(data_test['r_test'], dtype=torch.float32).to(self.device)
        self.mask_test = torch.tensor(data_test['mask_test'], dtype=torch.float32).to(self.device)

        self.train_num = self.x_train.shape[0]
        self.validation_num = self.x_validation.shape[0]
        self.test_num = self.x_test.shape[0]

        self.train_r_max = torch.max(self.r_train)
        self.train_r_min = torch.min(self.r_train)
