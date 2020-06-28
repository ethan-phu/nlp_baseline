from data import Wiki_Dataset
from model import glove_model
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import config as argumentparser
config = argumentparser.ArgumentParser()
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
wiki_dataset = Wiki_Dataset(min_count=config.min_count,window_size=config.window_size)
training_iter = torch.utils.data.DataLoader(dataset=wiki_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=2)
model = glove_model(len(wiki_dataset.word2id),config.embed_size,config.x_max,config.alpha)
if config.cuda and torch.cuda.is_available():
    torch.cuda.set_device(config.gpu)
    model.to(torch.device('cuda:0'))
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
loss= -1
for epoch in range(config.epoch):
    process_bar = tqdm(training_iter)
    for data, label in process_bar:
        w_data = torch.Tensor(np.array([sample[0] for sample in data])).long()
        v_data = torch.Tensor(np.array([sample[1] for sample in data])).long()
        if config.cuda and torch.cuda.is_available():
            w_data = w_data.cuda()
            v_data = v_data.cuda()
            label = label.cuda()
        loss_now =model(w_data,v_data,label)
        if loss==-1:
            loss = loss_now.data.item()
        else:
            loss = 0.95*loss+0.05*loss_now.data.item()
        process_bar.set_postfix(loss=loss)
        process_bar.update()
        optimizer.zero_grad()
        loss_now.backward()
        optimizer.step()
model.save_embedding(wiki_dataset.word2id,"./embeddings/result.txt")
