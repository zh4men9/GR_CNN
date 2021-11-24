# Author: Little-Chen
# Emial: Chenxiuyan_t@163.com

import os

def check_args(args): # 检测文件
    # results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass

    return args