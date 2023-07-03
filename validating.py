import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data

from utils import DrugTargetDataset, collate, ci,getdata_from_csv
from GTrans import GTrans1

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=64, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', default='davis', help='dataset: davis or kiba')
parser.add_argument('--testing-dataset-path', default='data/davis_test.csv',help='training dataset path: davis or kiba/ 5-fold or not')

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()
batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay


testing_dataset_address = args.testing_dataset_path

# processing testing data

test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address)
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity,pid)
dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#model
model = GTrans1()

model.load_state_dict(torch.load('saved_models/DAT_best_davis.model'),strict=False)

load_dict = torch.load('saved_models/DAT_best_davis.model')
model.eval()
if use_cuda:
    # model = nn.DataParallel(model)
    model.cuda()

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=lr)
criterion = nn.MSELoss()

test_epoch_size = len(test_drug)

print('--- GAT model --- ')

best_ci = 0


b = 0
total_loss = []
total_ci = []
total_pred = torch.Tensor()
total_label = torch.Tensor()

with torch.no_grad():
    for protein, d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, affinity in dataloader_test:

        if use_cuda:
            protein = [p.cuda() for p in protein]
            d_node = d_node.cuda()
            d_attn_bias = d_attn_bias.cuda()
            d_spatial_pos = d_spatial_pos.cuda()
            d_in_degree = d_in_degree.cuda()
            d_out_degree = d_out_degree.cuda()
            affinity = affinity.cuda()

        out,a,b = model(protein, d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree)

        loss = criterion(out, affinity)

        out = out.cpu()
        affinity = affinity.cpu()
        loss = loss.cpu().detach()
        c_index = ci(affinity.detach().numpy(), out.detach().numpy())

        b = b + batch_size
        total_loss.append(loss)
        total_ci.append(c_index)
        total_pred = torch.cat((total_pred, out), 0)
        total_label = torch.cat((total_label, affinity), 0)



all_ci = ci(total_label.detach().numpy().flatten(), total_pred.detach().numpy().flatten())
print('loss={:.5f}, ci={:.5f}\n'.format(np.mean(total_loss), all_ci))






