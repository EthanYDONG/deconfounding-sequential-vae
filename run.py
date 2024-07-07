import configs
import os
import torch
from model_decon import ModelDecon
from data_handler import DataHandler

def main():
    opts = configs.model_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('starting processing data ...')

    data = DataHandler(opts)

    print('starting initialising model ...')
    opts['r_range_upper'] = data.train_r_max
    opts['r_range_lower'] = data.train_r_min
    model = ModelDecon(opts).to(device)

    print('starting training model ...')
    model.train_model(data)

if __name__ == "__main__":
    main()
