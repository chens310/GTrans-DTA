import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from utils import DrugTargetDataset, collate, ci ,mse,getdata_from_csv,get_rm2
from GTrans import GTrans1
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=32, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
parser.add_argument('--training-dataset-path', default='data/davis_fold_3_train.csv', help='training dataset path: davis or kiba/ 5-fold or not')
parser.add_argument('--testing-dataset-path', default='data/davis_fold_3_valid.csv', help='training dataset path: davis or kiba/ 5-fold or not')

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay







training_dataset_address = args.training_dataset_path
testing_dataset_address = args.testing_dataset_path

#processing training data

train_drug, train_protein, train_affinity, pid,drug_id = getdata_from_csv(training_dataset_address)
train_affinity = torch.from_numpy(np.array(train_affinity)).float()


dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid,drug_id)
dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#processing testing data

test_drug, test_protein, test_affinity, pid,drug_id = getdata_from_csv(testing_dataset_address)

test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity, pid,drug_id)
dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#model
model = GTrans1()

# model.load_state_dict(torch.load('saved_models/davis_fold3.model'),strict=False)



if use_cuda:
    # model = nn.DataParallel(model)
    model.cuda()

    
#optimizer

optim = torch.optim.RAdam(model.parameters(), lr=lr,betas=(0.9, 0.999),eps=1e-08, weight_decay=weight_decay)
criterion = nn.MSELoss()


train_epoch_size = len(train_drug)
test_epoch_size = len(test_drug)

print('--- GTrans model --- ')
start =datetime.datetime.now()
print(start)
best_ci = 0
best_mse = 100000

for epoch in range(epochs):
    
    #train
    model.train()
    b = 0
    total_loss = []
    total_ci = []
    LOG_INTERVAL =50
    for protein,p_bias, d_node, d_attn_bias, d_spatial_pos,d_geo_dist, d_in_degree, d_out_degree, affinity in dataloader_train:

        if use_cuda:
            protein = [p.cuda() for p in protein]
            p_bias = [p.cuda() for p in p_bias]
            d_node = d_node.cuda()
            d_attn_bias = d_attn_bias.cuda()
            d_spatial_pos = d_spatial_pos.cuda()
            d_geo_dist = d_geo_dist.cuda()
            d_in_degree = d_in_degree.cuda()
            d_out_degree = d_out_degree.cuda()
            affinity = affinity.cuda()

        out, d_attn, p_attn = model(protein,p_bias, d_node, d_attn_bias, d_spatial_pos,d_geo_dist, d_in_degree, d_out_degree)
        loss = criterion(out, affinity)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        out = out.cpu()
        affinity = affinity.cpu()
        loss = loss.cpu().detach()
        c_index = ci(affinity.detach().numpy(),out.detach().numpy())

        b = b + batch_size
        total_loss.append(loss)
        total_ci.append(c_index)
        if b % LOG_INTERVAL == 0:
            print('# [{}/{}] training {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                        , epochs
                                                                        , b/train_epoch_size
                                                                        , loss
                                                                        , c_index


                         , end='\r'))
    
    print('total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), np.mean(total_ci)))
    
    model.eval()
    b=0
    total_loss = []
    total_ci = []
    total_pred = torch.Tensor()
    total_label = torch.Tensor()
    LOG_INTERVAL=50

    with torch.no_grad():
        for protein,p_bias, d_node, d_attn_bias, d_spatial_pos,d_geo_dist, d_in_degree, d_out_degree, affinity in dataloader_test:

            if use_cuda:
                protein = [p.cuda() for p in protein]
                p_bias = [p.cuda() for p in p_bias]
                d_node = d_node.cuda()
                d_attn_bias = d_attn_bias.cuda()
                d_spatial_pos = d_spatial_pos.cuda()
                d_geo_dist = d_geo_dist.cuda()
                d_in_degree = d_in_degree.cuda()
                d_out_degree = d_out_degree.cuda()
                affinity = affinity.cuda()

            out, test_attn, test_attn1 = model(protein,p_bias, d_node, d_attn_bias, d_spatial_pos,d_geo_dist, d_in_degree, d_out_degree)
            
            loss = criterion(out, affinity)
            
            out = out.cpu()
            affinity = affinity.cpu()
            loss = loss.cpu().detach()
            c_index = ci(affinity.detach().numpy(),out.detach().numpy())
            b = b + batch_size
            total_loss.append(loss)
            total_ci.append(c_index)
            total_pred = torch.cat((total_pred, out), 0)
            total_label = torch.cat((total_label, affinity), 0)
            if b % LOG_INTERVAL == 0:
                print('# [{}/{}] testing {:.1%} loss={:.5f}, ci={:.5f}\n'.format(epoch+1
                                                                            , epochs
                                                                            , b/test_epoch_size
                                                                            , loss
                                                                            , c_index

                                                                            )
                , end='\r')
    
    

    all_ci = ci(total_label.detach().numpy().flatten(),total_pred.detach().numpy().flatten())
    all_mse = mse(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
    rm2 = get_rm2(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
    print('total_loss={:.5f}, total_ci={:.5f}\n'.format(np.mean(total_loss), all_ci))
    save_path = 'saved_models/davis3d.model'
    end1=datetime.datetime.now()
    print("each_epoch_time:",end1-start)
    if all_ci > best_ci:

        best_ci = all_ci
        print('improve at this epoch ci={:.5f}'.format(best_ci))
        print('mse:',all_mse)
        print('rm2:',rm2)
        best_rm2=rm2
        best_mse =all_mse
        model.cpu()
        # save_dict = {'model':model.state_dict(), 'optim':optim.state_dict(), 'ci':best_ci}
        torch.save(model.state_dict(), save_path)
        if use_cuda:
            model.cuda()
    else:
        print("no improve at this epoch")
        print("the best ci and mse,rm2",best_ci,best_mse,best_rm2)

end2=datetime.datetime.now()
print("total_time:",end2-start)







      