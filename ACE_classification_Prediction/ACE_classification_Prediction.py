from pyuul import VolumeMaker
from pyuul import utils
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
import shutil
import MDAnalysis as mda
import subprocess
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset,DataLoader
import pandas as pd
device = torch.device("cuda") 
import os
join=os.path.join
from transformers import AutoTokenizer
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import OrderedDict
from tqdm import tqdm

model_checkpoint = "facebook/esm2_t6_8M_UR50D"
pdb_path = "./structure/"
seq_path = "./test.csv"
temp_path = "./temp/"
save_path = "./output.csv"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(4)

batch_size = 1
num_labels = 2
radius = 2
n_features = 1024
hid_dim = 300
n_heads = 1
dropout = 0.1

class MyDataset(Dataset):
    def __init__(self,dict_data) -> None:
        super(MyDataset,self).__init__()
        self.data=dict_data
        self.structure=pdb_structure(dict_data['structure'])
    def __getitem__(self, index):
        return self.data['text'][index], self.structure[index]
    def __len__(self):
        return len(self.data['text'])
        
def collate_fn(batch):
    data = [item[0] for item in batch]
    structure = torch.tensor([item[1].tolist() for item in batch]).to(device)
    max_len = max([len(b[0]) for b in batch])+2
    fingerprint = torch.tensor(peptides_to_fingerprint_matrix(data, radius, n_features),dtype=float).to(device)
    pt_batch=tokenizer(data, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    return {'input_ids':pt_batch['input_ids'].to(device),
            'attention_mask':pt_batch['attention_mask'].to(device)}, structure, fingerprint

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super(CrossAttentionBlock, self).__init__()
        self.att = AttentionBlock(hid_dim = hid_dim, n_heads = n_heads, dropout=dropout)    
    def forward(self, structure_feature, fingerprint_feature, sequence_feature):
        # cross attention for compound information enrichment
        fingerprint_feature = fingerprint_feature + self.att(fingerprint_feature, structure_feature, structure_feature)
        # self-attention
        fingerprint_feature = self.att(fingerprint_feature, fingerprint_feature, fingerprint_feature)
        # cross-attention for interaction
        output = self.att(fingerprint_feature, sequence_feature, sequence_feature)
        return output
    
def peptides_to_fingerprint_matrix(peptides, radius=radius, n_features=n_features):
    n_peptides = len(peptides)
    features = np.zeros((n_peptides, n_features))
    for i, peptide in enumerate(peptides):
        mol = Chem.MolFromSequence(peptide)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_features)
        fp_array = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
        features[i, :] = fp_array
    return features

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=hid_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(300,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc_fingerprint = nn.Linear(1024,hid_dim)
        self.fc_structure = nn.Linear(1500,hid_dim)
        self.fingerprint_lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=1024, hidden_size=1024//2, batch_first=True)
        self.structure_lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=500, hidden_size=500//2, batch_first=True)
        self.output_layer = nn.Linear(64,num_labels)
        self.dropout = nn.Dropout(0)
        self.CAB = CrossAttentionBlock()
    def forward(self,structure, x, fingerprint):
        fingerprint = torch.unsqueeze(fingerprint, 2).float()
        structure = structure.permute(0, 2, 1)
        fingerprint = fingerprint.permute(0, 2, 1)
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'].to(device),attention_mask=x['attention_mask'].to(device)) 
        sequence_feature = self.dropout(bert_output["logits"])
        structure = structure.to(device)
        fingerprint_feature, _  = self.fingerprint_lstm(fingerprint)
        structure_feature, _  = self.structure_lstm(structure)
        fingerprint_feature = fingerprint_feature.flatten(start_dim=1)
        structure_feature = structure_feature.flatten(start_dim=1)
        fingerprint_feature = self.fc_fingerprint(fingerprint_feature)
        structure_feature = self.fc_structure(structure_feature)
        output_feature = self.CAB(structure_feature, fingerprint_feature, sequence_feature)
        output_feature = self.dropout(self.relu(self.bn1(self.fc1(output_feature))))
        output_feature = self.dropout(self.relu(self.bn2(self.fc2(output_feature))))
        output_feature = self.dropout(self.relu(self.bn3(self.fc3(output_feature))))
        output_feature = self.dropout(self.output_layer(output_feature))
        return torch.softmax(output_feature,dim=1)

def pdb_structure(Structure_index):
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

    os.makedirs(temp_path)

    for i in Structure_index:
        old_path = join(pdb_path,i+'.pdb')
        new_path = join(temp_path,i+'.pdb')
        shutil.copy2(old_path, new_path)
    coords, atname, pdbname, pdb_num = utils.parsePDB(temp_path)
    # pdbname = [x.split(".")[0] for x in pdbname]

    atoms_channel = utils.atomlistToChannels(atname)
    radius = utils.atomlistToRadius(atname)

    PointCloudSurfaceObject = VolumeMaker.PointCloudVolume(device=device)

    coords = coords.to(device)  
    radius = radius.to(device) 
    atoms_channel = atoms_channel.to(device) 

    SurfacePoitCloud = PointCloudSurfaceObject(coords, radius) 
 
    return SurfacePoitCloud

if __name__ == '__main__':
    df = pd.read_csv(seq_path)
    test_sequences = df["Seq"].tolist()
    test_Structure_index = df["Structure_index"].tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    test_dict = {"text":test_sequences, 'structure':test_Structure_index}
    test_data=MyDataset(test_dict)
    test_dataloader=DataLoader(test_data,batch_size=batch_size,collate_fn=collate_fn,shuffle=False)

    model = MyModel()
    model.load_state_dict(torch.load("best_model.pth"))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        probability_all = []
        Target_all = []
        print("=================================Start prediction========================")
        for index, (batch, structure_fea, fingerprint) in enumerate(test_dataloader):
            batchs = {k: v for k, v in batch.items()}
            outputs = model(structure_fea, batchs, fingerprint)
            probability = outputs[0].tolist()
            train_argmax = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            for j in range(0,len(train_argmax)):
                output = train_argmax[j]
                if output == 0:
                    Target = "low"
                    probability = probability[0]
                elif output == 1:
                    Target = "high"
                    probability = probability[1]
                print(Target, probability)
                probability_all.append(probability)
                Target_all.append(Target)

    summary = OrderedDict()
    summary['Seq'] = test_sequences
    summary['Target'] = Target_all
    summary['Probability'] = probability_all
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(save_path, index=False)