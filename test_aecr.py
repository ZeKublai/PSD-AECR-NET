import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from datasets.pretrain_datasets import TrainData, ValData, TestData, TestData2, TestData_GCA, TestData_FFA
from models.GCA import GCANet
from models.FFA import FFANet
from models.MSBDN import MSBDNNet
from models.AECRNet import Dehaze
from utils import to_psnr, print_log, validation, adjust_learning_rate
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    epoch = 20
    test_data_dir = 'images/cvpr/'
    
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = Dehaze(3, 3)
    net = nn.DataParallel(net, device_ids=device_ids)
    net.load_state_dict(torch.load('pre-trained/PSD-AECRNET-13'))
    net.eval()

    test_data_loader = DataLoader(TestData_FFA(test_data_dir), batch_size=1, shuffle=False, num_workers=8) # For FFA and MSBDN

    output_dir = 'output/AECRNET/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
        
    with torch.no_grad():
        for batch_id, val_data in enumerate(test_data_loader):
            if batch_id > 1:
                break
            haze, haze_A, name = val_data # For FFA and MSBDN

            if haze.size()[2] % 16 != 0 or haze.size()[3] % 16 != 0:
                haze = F.upsample(haze, [haze.size()[2] + 16 - haze.size()[2] % 16,
                                haze.size()[3] + 16 - haze.size()[3] % 16], mode='bilinear')
            if haze_A.size()[2] % 16 != 0 or haze_A.size()[3] % 16 != 0:
                haze_A = F.upsample(haze_A, [haze_A.size()[2] + 16 - haze_A.size()[2] % 16,
                                haze_A.size()[3] + 16 - haze_A.size()[3] % 16], mode='bilinear')

            haze.to(device)
            
            print(batch_id, 'BEGIN!')
            
            net.eval()
            _, pred, T, A, I = net(haze, haze_A, True) # For FFA and MSBDN
            
            ### FFA & MSBDN ###
            ts = torch.squeeze(pred.clamp(0, 1).cpu())
            vutils.save_image(ts, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
            torch.cuda.empty_cache()
            ###################
