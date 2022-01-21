import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from datasets.pretrain_datasets import TrainData, ValData, TestData, TestData2, TestData_GCA, TestData_FFA
from models.GCA import GCANet
from models.FFA import FFANet
from models.MSBDN import MSBDNNet
from utils import to_psnr, print_log, validation, adjust_learning_rate
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    epoch = 14
    test_data_dir = 'images/cvpr/'
    
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = GCANet(in_c=4, out_c=3, only_residual=True).to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    net.load_state_dict(torch.load('pre-trained/PSD-GCANET'))
    net.eval()

    test_data_loader = DataLoader(TestData_GCA(test_data_dir), batch_size=1, shuffle=False, num_workers=8)

    output_dir = 'output/GCA/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
        
    with torch.no_grad():
        for batch_id, val_data in enumerate(test_data_loader):
            if batch_id > 1:
                break
            haze, name = val_data # For GCA
            haze.to(device)

            if haze.size()[2] % 16 != 0 or haze.size()[3] % 16 != 0:
                haze = F.upsample(haze, [haze.size()[2] + 16 - haze.size()[2] % 16,
                                haze.size()[3] + 16 - haze.size()[3] % 16], mode='bilinear')
            
            print(batch_id, 'BEGIN!')
            
            net.eval()
            pred = net(haze, 0, True, False) # For GCA
            
            ### GCA ###
            dehaze = pred.float().round().clamp(0, 255)
            out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
            out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
            ###########
