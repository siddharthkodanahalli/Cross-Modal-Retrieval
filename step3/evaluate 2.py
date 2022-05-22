#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from PIL import Image

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
from tqdm import tqdm

import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from time import sleep
import random


# In[2]:


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

image_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


# In[3]:


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


# In[4]:


class MyModel(nn.Module):
    def __init__(self,device, d_model, concat_size, num_classes, nlayers, nhead, d_hid, dropout=0.5):
        
        super(MyModel, self).__init__()
        
        self.device = device
        
        
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
        
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.class_token = nn.Parameter(torch.rand(1, d_model))
        
        self.linear_layer = nn.Sequential(nn.Linear(concat_size,1), nn.Sigmoid())
    
    def preprocess(self,image, encoded_text_input):
        
        #input_img = self.vit_feature_extractor(image, return_tensors='pt').to(device)

        with torch.no_grad():
            text_embeds = self.bert_model.embeddings(input_ids = encoded_text_input['input_ids'], 
                                            token_type_ids = encoded_text_input['token_type_ids']).to(device)
            image_embeds = self.vit_model.embeddings(image['pixel_values']).to(device)

        return image_embeds, text_embeds
        
    def forward(self, images, texts):
        image_patch_embeds, text_embeddings = self.preprocess(images, texts)
        
        concat_embeds = torch.cat([image_patch_embeds, text_embeddings], 1)
        concat_embeds = torch.stack([torch.vstack((self.class_token, concat_embeds[i])) for i in range(len(concat_embeds))])
        pos_embed = get_positional_embeddings(concat_embeds.shape[1], self.d_model).repeat(concat_embeds.shape[0], 1, 1)
        concat_embeds+= pos_embed.to(self.device)
        
        logits = self.transformer_encoder(concat_embeds)
        logits = logits[:,0,:]
        
        preds = self.linear_layer(logits)
        
        return preds


# In[5]:


def preprocess(image, text):
    encoded_text_input = bert_tokenizer(text, return_tensors='pt',padding='max_length', truncation=True).to(device)
    input_img = image_feature_extractor(image, return_tensors='pt').to(device)

    return input_img, encoded_text_input


# In[7]:


device = "cuda:1" if torch.cuda.is_available() else "cpu"


# In[9]:


class MyDataset(Dataset):
    
    def __init__(self, annotations, img_dir):
        self.labels = pd.read_csv(annotations)
        self.img_dir = img_dir 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx,1]
        img_path = self.img_dir + img_name[0] + "/" + \
                img_name[1] + "/" + img_name[2] + "/" + img_name[3] + "/" + img_name
        
        im = read_image(img_path)
        
        recipe = self.labels.iloc[idx,2]
        
        label = self.labels.iloc[idx, 3]
        
        #t_embeds, img_embeds = self.preprocess(im, recipe)
        return im, recipe, label


# In[10]:


def collate_function(batch):
    imgs = []
    texts = []
    labels = []
    for img, text, label in batch:
        
        imgs.append(img)
        texts.append(text)
        labels.append(label)
    
    encoded_text_input = bert_tokenizer(texts, return_tensors='pt',padding='max_length', truncation=True).to(device)
    imgs_feats = image_feature_extractor(imgs, return_tensors='pt')
    labels = torch.tensor(labels)
    return imgs_feats,encoded_text_input,labels
    
        


# In[11]:


test_d = MyDataset('sampled_100_test.csv','/freespace/local/sk2381/im2recipe-Pytorch/data/test/')


# In[17]:


pd.read_csv('test.csv')


# In[12]:


from torch.utils.data import DataLoader
test_dataloader = DataLoader(test_d, collate_fn=collate_function,batch_size=64, shuffle=False)


# In[8]:


transformer_model = MyModel(device, 768,768,2,1,2,512,0.5).to(device)
checkpoint = torch.load('models_ins2/model.pt')
transformer_model.load_state_dict(checkpoint['model_state_dict'])


# In[32]:


def test(dataloader, model):
    i=0
    preds_list = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for img,recipe, y in tepoch:
            tepoch.set_description(f"Epoch {i}")
            img = img.to(device)
            recipe = recipe.to(device)
            with torch.no_grad():
                pred = model(img, recipe)
            preds_list.extend(pred)
            i+=1
    return preds_list
            


# In[33]:


preds_final = test(test_dataloader,transformer_model)


# In[36]:


test_df = pd.read_csv('test.csv')


# In[38]:


test_df['predictions'] = torch.cat(preds_final).cpu().numpy()


# In[39]:


test_df.to_csv('final_predictions.csv')


# In[ ]:




