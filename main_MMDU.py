import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

from pretraineddataset import PretrainedDataset
from mmdu_utils import MatConvert, MMDu, TST_MMD_u, ModelLatentF
from mae_model import Get_MAE, MAE_encoder
from torchvision import transforms
from covidxdataset import COVIDxDataset
from cxrdataset import init_CXR
from torch.utils.data import DataLoader



###
def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.dc_aug_param = None
    args.method = 'DM'
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        print(f'    Using {len(images_all)} data')



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        def get_images_class_indep(n): # get random n images that are class independent
            idx_shuffle = np.random.permutation(np.arange(len(images_all)))[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))



        '''MMD pre-training'''

        # Load pretrained MAE model
        mae_net = Get_MAE()
        mae_net.load_state_dict(torch.load('mae_covidx.pt', map_location=torch.device(args.device)))
        mae_net_encoder = MAE_encoder(mae_net.encoder)
        mae_net_encoder = mae_net_encoder.to(args.device)
        mae_net_encoder.train()
        for param in list(mae_net_encoder.parameters()):
            param.requires_grad = False
        print('Sucessfully load MAE encoder')

        dtype = torch.float
        N_per = 200 # permutation times
        d = 1024
        n = 50
        x_in = d # number of neurons in the input layer, i.e., dimension of data
        H =3*d # number of neurons in the hidden layer
        x_out = 3*d # number of neurons in the output layer
        learning_rate = 0.00005 # default learning rate for MMD-D on HDGM
        N_epoch = 100 # number of training epochs
        
        model_u = ModelLatentF(x_in, H, x_out).to(args.device)
        
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), args.device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * d), args.device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), args.device, dtype)
        sigma0OPT.requires_grad = True

        # Setup optimizer for training deep kernel
        optimizer_u = torch.optim.Adam(list(model_u.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT],
                                    lr=learning_rate)
        s1 = mae_net_encoder(get_images_class_indep(n))[0]
        s2 = mae_net_encoder(get_images_class_indep(n))[0]
        S = torch.cat((s1, s2), dim=0).to(args.device)
        # S = MatConvert(S, args.device, dtype)
        N1 = len(S)//2


        # Get data from CovidX and ChildXRay for training MMD on different distributions
        # CovidX
        im_size = (224, 224)
        covidx_mean = [0.4886, 0.4886, 0.4886]
        covidx_std = [0.2460, 0.2460, 0.2460]
        covidx_transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(), 
            transforms.Normalize(covidx_mean, covidx_std)
            ])
        covidx_train_ = COVIDxDataset(transform=covidx_transform, flag='train') 
        # ChildXRay
        cxr_train_ = init_CXR(mode='train') 
        # Subset
        train_length = min(len(covidx_train_), len(cxr_train_))
        rand_idx = np.random.randint(0, train_length, train_length)
        covidx_train = torch.utils.data.Subset(covidx_train_, rand_idx)
        cxr_train = torch.utils.data.Subset(cxr_train_, rand_idx)
        covidx_loader = DataLoader(covidx_train, batch_size=n, shuffle=True, num_workers=4)
        cxr_loader = DataLoader(cxr_train, batch_size=n, shuffle=True, num_workers=4)

        # Train deep kernel to maximize test power
        np.random.seed(seed=1102)
        torch.manual_seed(1102)
        torch.cuda.manual_seed(1102)
        mmds___ = []
        print('\n----------Begin Training MMD kernel----------')
        for t in range(N_epoch):
            for covidx_data, cxr_data in zip(covidx_loader, cxr_loader):
                covidx_images, _ = covidx_data
                cxr_images, _ = cxr_data
                covidx_images = covidx_images.to(args.device)
                cxr_images = cxr_images.to(args.device)
                s1_ = mae_net_encoder(covidx_images)[0]
                s2_ = mae_net_encoder(cxr_images)[0]
                S_ = torch.cat((s1_, s2_), dim=0).to(args.device)
                N_ = len(S_)//2
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute output of the deep network
                modelu_output = model_u(S_)
                # Compute J (STAT_u)
                TEMP = MMDu(modelu_output, N_, S_, sigma, sigma0_u, ep)
                mmd_value_temp = -(TEMP[0]+10**(-8))
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                if mmd_std_temp.item() == 0:
                    print('error!!')
                if np.isnan(mmd_std_temp.item()):
                    print('error!!')
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                # Initialize optimizer and Compute gradient
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                # Update weights using gradient descent
                optimizer_u.step()
                # Print MMD, std of MMD and J
            if t % 10 ==0:
                print("mmd_value: ", -1 * mmd_value_temp.item(), "mmd_std: ", mmd_std_temp.item(), "Statistic: ",
                    -1 * STAT_u.item())
                mmds___.append(-1 * mmd_value_temp.item())
        
        h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, 0.5, args.device, dtype)
        print("MMD_value:", mmd_value_u)
        torch.save(model_u.state_dict(), './pretrained/modelu_xray.pt')
        print(f'Saved model in ./pretrained/modelu_xray.pt')
        print('----------Finish Training MMDu kernel----------\n')

        model_u.train()
        for param in list(model_u.parameters()):
            param.requires_grad = False
        epsilonOPT.requires_grad = False
        sigmaOPT.requires_grad = False
        sigma0OPT.requires_grad = False



        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        elif args.init == 'MAE_CovidX':
            print('initialize synthetic data from pretrained CovidX images')
            pretrained_set = PretrainedDataset(dataset='MAE_CovidX', ipc = 50, im_size = [224, 224])
            images_pretrained_set = [torch.unsqueeze(pretrained_set[i], dim=0) for i in range(len(pretrained_set))]
            images_pretrained_set = torch.cat(images_pretrained_set, dim=0).to(args.device)
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = images_pretrained_set[c*args.ipc:(c+1)*args.ipc].detach().data
        else:
            print('initialize synthetic data from random noise')



        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'modelu_output_%s_iter%d.png'%(args.dataset, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            # net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            # net.train()
            # for param in list(net.parameters()):
            #     param.requires_grad = False

            # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
            # embed = mae_net_encoder.module if torch.cuda.device_count() > 1 else mae_net_encoder # for GPU parallel

            loss_avg = 0

            ''' update synthetic data '''
            torch.autograd.set_detect_anomaly(True)
            if 'BN' not in args.model: # for ConvNet
            # if False:
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.ipc)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    # output_real = embed(img_real)[1].detach()
                    # output_syn = embed(img_syn)[1]

                    # loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                    s1 = mae_net_encoder(img_real)[0]
                    s2 = mae_net_encoder(img_syn)[0]
                    S = torch.cat((s1, s2), dim=0).to(args.device)
                    # S = MatConvert(S, args.device, dtype)
                    N1 = len(S)//2
                    
                    # h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, 0.5, args.device, dtype)
                    # loss += mmd_value_u

                    ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                    sigma = sigmaOPT ** 2
                    sigma0_u = sigma0OPT ** 2
                    # Compute output of the deep network
                    modelu_output = model_u(S)
                    TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
                    mmd_value_temp = (TEMP[0]+10**(-8))
                    mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                    if mmd_std_temp.item() == 0:
                        print('error!!')
                    if np.isnan(mmd_std_temp.item()):
                        print('error!!')
                    STAT_u = torch.div(mmd_value_temp, mmd_std_temp) * 100
                    loss += STAT_u

                    

            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.ipc)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                # output_real = embed(images_real_all)[1].detach()
                # output_syn = embed(images_syn_all)[1]

                # loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

                s1 = mae_net_encoder(img_real)[0]
                s2 = mae_net_encoder(img_syn)[0]
                S = torch.cat((s1, s2), dim=0).to(args.device)
                # S = MatConvert(S, args.device, dtype)
                N1 = len(S)//2

                # h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, N1, S, sigma, sigma0_u, ep, 0.5, args.device, dtype)
                # loss += mmd_value_u

                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute output of the deep network
                modelu_output = model_u(S)
                TEMP = MMDu(modelu_output, N1, S, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0]+10**(-8))
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                if mmd_std_temp.item() == 0:
                    print('error!!')
                if np.isnan(mmd_std_temp.item()):
                    print('error!!')
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                loss += STAT_u

            # print(f'loss: {loss.item()}')
            optimizer_img.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (num_classes)

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
    print(mmds___)



if __name__ == '__main__':
    main()


