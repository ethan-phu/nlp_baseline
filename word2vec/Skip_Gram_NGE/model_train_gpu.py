import sys
sys.path.append("../Skip_Gram_NGE/")
from skip_gram_nge_model import SkipGramModel
from input_data import InputData
import torch.optim as optim
from tqdm import tqdm
import torch
import argumentparser as argumentparser
args = argumentparser.ArgumentParser()
WINDOW_SIZE = args.window_size  # 上下文窗口c
BATCH_SIZE = args.batch_size  # mini-batch
MIN_COUNT = args.min_count  # 需要剔除的 低频词 的频
EMB_DIMENSION = args.embed_dimension  # embedding维度
LR = args.learning_rate  # 学习率
NEG_COUNT = args.neg_count  # 负采样数


class Word2Vec:
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = InputData(input_file_name, MIN_COUNT)
        self.model = SkipGramModel(self.data.word_count, EMB_DIMENSION).cuda()
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        #self.model.load_state_dict(torch.load("../results/skipgram_nge.pkl"))
        print("SkipGram Training......")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(5*batch_count)))
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_w = [int(pair[0]) for pair in pos_pairs]
            pos_v = [int(pair[1]) for pair in pos_pairs]
            neg_v = self.data.get_negative_sampling(pos_pairs, NEG_COUNT)
            pos_w = pos_w
            pos_v = pos_v
            neg_v = neg_v

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_w, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()
            process_bar.set_postfix(loss=loss.data)
            process_bar.update()
        torch.save(self.model.state_dict(),"../results/skipgram_nge.pkl")
        self.model.save_embedding(self.data.id2word_dict, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='../data/wiki_data.txt', output_file_name="../results/skip_gram_neg.txt")
    w2v.train()