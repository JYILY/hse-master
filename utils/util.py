import argparse
import logging
import os
import time
import shutil

def make_save_path(name):
    save_path = "trains\\" + "<" + time.strftime("%m-%d %H-%M", time.localtime()) + ">\\"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    return save_path

def make_logger(save_path,name):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(save_path + "train-result.log")
    console = logging.StreamHandler()
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def get_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)