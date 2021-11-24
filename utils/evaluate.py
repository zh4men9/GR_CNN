# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com

import torch
import logging

def evaluate(model, device, test_dataloader):
    correct_class = 0    
    n = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            n += target.shape[0]

            model_output = model(data)
            
            pred_class = model_output.argmax(dim=1, keepdim=True)
            
            correct_class += pred_class.eq(target.view_as(pred_class)).sum().item()

            logging.info('Test ACC: {}/{} ({:.3f}%)'.format(correct_class, n, 100*correct_class/float(n)))

    return 100. * correct_class / float(n)