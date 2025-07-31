from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
import sys
import torch
torch.manual_seed(0)

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu

from fedlab.utils.dataset import MNISTPartitioner,CIFAR10Partitioner

import pickle
from gzip import compress,decompress

from enhancedcore import ModifiedSubsetSerialTrainer
from models.cnn_mnist import CNNMNIST
from models.cnn_cifar import CNNCifar
from utils import *
from pruning_utils import *
from conf import *

# configuration
parser = argparse.ArgumentParser(description="FedLTH Standalone Mode")
# Server Global Config
parser.add_argument("--total_client", type=int, default=10)
parser.add_argument("--com_round", type=int, default=100)
parser.add_argument("--pretrained",type=str,default=None)

# Client Config - Dataset
parser.add_argument("--dataset", type=str, default="mnist")
#parser.add_argument("--balance", type=bool, default=True)
parser.add_argument("--niid", type=bool, default=False)

# Client Config - Sample
parser.add_argument("--sample_ratio", type=float,default=1)

# Client Config - Train params
parser.add_argument("--batch_size", type=int,default=128)
parser.add_argument("--epochs", type=int,default=5)
parser.add_argument("--cuda", type=bool, default=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_args()

if __name__ == "__main__":
    if args.dataset == 'mnist':
        root="./dataset/mnist/"
        trainset = torchvision.datasets.MNIST(root=root,
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
        trainset_partition = MNISTPartitioner(trainset.targets,
                                      args.total_client,
                                      partition="noniid-labeldir" if args.niid else "iid",
                                      dir_alpha=dir_alpha,
                                      seed=seed)
        testset = torchvision.datasets.MNIST(root=root,
                                     train=False,
                                     download=True,
                                     transform=transforms.ToTensor())

        
        model = CNNMNIST()
    elif args.dataset == 'cifar10':
        root="./dataset/cifar10/"
        trainset = torchvision.datasets.CIFAR10(root=root,
                                                train=True,
                                                download=True,
                                                transform=transforms.ToTensor())
        trainset_partition = CIFAR10Partitioner(trainset.targets,
                                      args.total_client,
                                      #balance=args.balance,
                                      partition="dirichlet" if args.niid else "iid",
                                      dir_alpha=dir_alpha,
                                      seed=seed)
        testset = torchvision.datasets.CIFAR10(root=root,
                                        train=False,
                                        download=True,
                                        transform=transforms.ToTensor())
        model = CNNCifar()
        #model = CNN_CIFAR10()
    else:
        raise ValueError("Invalid dataset:", args.dataset)
    
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=len(testset),
                                                drop_last=False,
                                                shuffle=False,
                                                # num_workers=8
                                                )
    
    train_loader = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.batch_size,
                                                drop_last=False,
                                                shuffle=True)


    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
        print(f"Pretrained Model Loaded {args.pretrained}")

    if args.cuda and torch.cuda.is_available():
        gpu = get_best_gpu()
        print("Using GPU:", gpu)
        model = model.cuda(gpu)
    else:
        print("Using CPU")
        args.cuda = False

    trainer = ModifiedSubsetSerialTrainer(model=model,
                              dataset=trainset,
                              data_slices=trainset_partition,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "lr": lr
                              },
                              cuda=args.cuda,
                              gpu=gpu if args.cuda else None)
    
    
    # FL Train Configs
    num_per_round = int(args.total_client * args.sample_ratio)
    aggregator = Aggregators.fedavg_aggregate

    to_select = [i for i in range(args.total_client)]
    selection = random.sample(to_select, num_per_round)
    
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print(f'Initial Accuracy: {acc}')
    lr_update = LR_UPDATE(init_lr=lr,min_lr=min_lr,decrease_rate=decrease_rate,decrease_frequency=decrease_frequency) ### 也许只在客户端用scheduler更好？

    pruner=FedLTHPruner(start_ratio=start_ratio,end_ratio=end_ratio,device='cpu' if args.cuda==False else gpu,channel_sparsity=channel_sparsity,min_inscrease=min_inscrease)
    
    acc_list =[acc]
    sp_round=0

    # Train Loop
    early_train_phase = True if acc <= prune_threshold else False
    for round in range(args.com_round):
        if early_train_phase and acc > prune_threshold:
            early_train_phase = False
            print(f"Accuracy Reached Prune Threshold({prune_threshold}), Early Prune Phase End")
        train_selection = to_select if early_train_phase else selection ### 尽量不在剪枝过程中切换客户端组
        trainer.args["lr"]=lr_update()

        model_parameters = SerializationTool.serialize_model(model)
        parameters_list = trainer.local_process(model_data=[model_parameters],
                                                id_list=train_selection)
        #parameters_list = trainer.local_process(model_data=compress(pickle.dumps(model)),
        #                                        id_list=train_selection)
        print(f"Transfered Size: {len(model_parameters)}")
        SerializationTool.deserialize_model(model, aggregator(parameters_list))

        criterion = nn.CrossEntropyLoss()
        loss, acc = evaluate(model, criterion, test_loader)
        acc_list.append(acc)
        print("EPOCH {}/{}: global lr: {:.4f}, loss: {:.4f}, acc: {:.2f}".format(round+1,args.com_round,trainer.args["lr"],loss, acc))

        if early_train_phase==False and sp_round < max_sp_rounds:
            if round % prune_step == 0:
                pruner.unstuctured_prune(model,conv1=False)
                weight_with_mask = deepcopy(model.state_dict())
                pruner.remove_prune(model, conv1=False)
                print(f'Prune Phase {sp_round+1}/{max_sp_rounds}[Unstruct Prune]: Prune Ratio: {pruner.ratio},Target Ratio:{pruner.end_ratio}')

            #剪枝后的一轮，如果精度下降超过5%则增大剪枝间隔
            if round % prune_step == 1:
                if acc_list[-2]-acc_list[-1]>0.05:
                    prune_step+=2
            
            if pruner.ratio >= pruner.end_ratio and pruner.unpruned_flag:
                num_classes=len(trainset.classes)
                new_model, sparsity = pruner.structured_prune(model,weight_with_mask,train_loader,criterion,num_classes)
                model = new_model
                print(f'Prune Phase {sp_round+1}/{max_sp_rounds}[Refill Struct Prune]: Final sparsity:' + str(100 * sparsity) + '%')
                
                trainer = ModifiedSubsetSerialTrainer(model=model, # Use the structurally pruned model
                                          dataset=trainset,
                                          data_slices=trainset_partition,
                                          args={
                                              "batch_size": args.batch_size,
                                              "epochs": args.epochs,
                                              "lr": lr
                                          },
                                          cuda=args.cuda,
                                          gpu=gpu if args.cuda else None)

                selection = random.sample(to_select, num_per_round)
                lr_update(reset=True)
                pruner.ratio = pruner.start_ratio
                pruner.unpruned_flag = False
                sp_round += 1
        
        if early_train_phase==False and sp_round >= max_sp_rounds:
            selection = random.sample(to_select, num_per_round)

    print("Finished Training")
    model_save_path = f"result/{model.__class__.__name__}_{args.dataset}_{args.com_round}rounds"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model, f"{model_save_path}/model.pth")
    print(f"Model saved to {model_save_path}")

            



    

