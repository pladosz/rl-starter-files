import csv
import os
import torch
import logging
import sys

import utils
from .other import device


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir,agent_id = 0, best = False, update = 0):
    if best == False:
        return os.path.join(model_dir, "status_agent_id_{0}.pt".format(agent_id))
    if best == True:
        return os.path.join(model_dir, "status_best_u_{0}.pt".format(update))


def get_status(model_dir, agent_id = 0, best = False, update = 0):
    path = get_status_path(model_dir, agent_id = agent_id, best = best, update = update)
    return torch.load(path, map_location=device)


def save_status(status, model_dir,agent_id = 0,best = False, update = 0):
    path = get_status_path(model_dir,agent_id = agent_id, best = best, update = update)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir, agent_id = 0, best = False, update = 0):
    return get_status(model_dir,agent_id = agent_id, best = best, update = update)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
