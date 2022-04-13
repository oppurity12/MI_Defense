from unicodedata import name
import torch, os, engine, classify, utils, sys
import numpy as np 
import torch.nn as nn
from sklearn.model_selection import train_test_split

from classify import VGG16

from engine import test

from torch.nn.utils.prune import l1_unstructured, ln_structured, remove

dataset_name = "celeba"
device = "cuda"
root_path = "./target_model"
log_path = os.path.join(root_path, "target_logs")
model_path = os.path.join(root_path, "target_ckp")
os.makedirs(model_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

def prune(model, amount):
    conv_lists = []
    cnt = 0

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            cnt += 1
            if cnt == 1:
                continue

            conv_lists.append(m)
    
    for m in conv_lists:
        l1_unstructured(m, name='weight', amount=amount)
        remove(m, 'weight')
    return model


def main(args, model_name, trainloader, testloader):
    net = VGG16(1000)
    path_net = './target_model/target_ckp/VGG16_86.87_pruned_0.20.tar'
    
    ckp_net = torch.load(path_net)
    dic = ckp_net['state_dict']
    tmp = {}

    for key, val in dic.items():
      tmp[key[7:]] = val
    
    net.load_state_dict(tmp, strict=True)


    criterion = nn.CrossEntropyLoss().cuda()

    print("Start Training!")

    amount = 0.4
    # prune(net, amount)

    net = torch.nn.DataParallel(net).cuda()

    accuracy =  test(net, criterion, testloader)

    print(f"accuracy: {accuracy:.3f}")
	
    # torch.save({'state_dict':net.state_dict()}, os.path.join(model_path, "{}_{:.2f}_pruned_{:.2f}.tar").format(model_name, accuracy, amount))

if __name__ == '__main__':
    file = "./config/classify.json"
    args = utils.load_json(json_file=file)
    model_name = args['dataset']['model_name']

    log_file = "{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    os.environ["CUDA_VISIBLE_DEVICES"] = args['dataset']['gpus']
    print(log_file)
    print("---------------------Training [%s]---------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name], dataset=args['dataset']['name'])

    train_file = args['dataset']['train_file_path']
    test_file = args['dataset']['test_file_path']
    _, trainloader = utils.init_dataloader(args, train_file, mode="train")
    _, testloader = utils.init_dataloader(args, test_file, mode="test")

    main(args, model_name, trainloader, testloader)