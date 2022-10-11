import os
import numpy as np
import torch
import argparse
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, ParamDiffAug, TensorDataset, get_daparam
import copy
import gc


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--num_eval', type=int, default=2, help='the number of evaluating randomly initialized models')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']

    saved_data = torch.load('./res_DC_{}_ConvNet_{}ipc.pt'.format(args.dataset, args.ipc))
    image_syn_eval = saved_data['data'][0][0]
    label_syn_eval = saved_data['data'][0][1]



    ''' Evaluate synthetic data '''
 
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))
        
        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
        print('DC augmentation parameters: \n', args.dc_aug_param)

        if args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
        else:
            args.epoch_eval_train = 300

        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))



if __name__ == '__main__':
    main()




