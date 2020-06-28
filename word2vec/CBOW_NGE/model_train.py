import sys
sys.path.append("../CBOW_NGE/")

from cbow_nge_model import CBOWModel
from input_data import InputData
import torch.optim as optim
from tqdm import tqdm
import argumentparser as argumentparser
args = argumentparser.ArgumentParser()
# hyper parameters
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
        self.model = CBOWModel(self.data.word_count, EMB_DIMENSION).cuda()
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        print("CBOW Training......")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))
        loss = -1
        for i in process_bar:
            pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_w = [int(pair[1]) for pair in pos_pairs]
            neg_w = self.data.get_negative_sampling(pos_pairs, NEG_COUNT)

            self.optimizer.zero_grad()
            loss_now = self.model.forward(pos_u, pos_w, neg_w)
            if loss==-1:
                loss = loss_now.data.item()
            else:
                loss = 0.95*loss+0.05*loss_now.data.item()
            loss_now.backward()
            self.optimizer.step()

            if i * BATCH_SIZE % 100000 == 0:
                self.lr = self.lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            process_bar.set_postfix(loss=loss)
            process_bar.update()

        self.model.save_embedding(self.data.id2word_dict, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='../data/wiki_data.txt', output_file_name="../results/cbow_neg.txt")
    w2v.train()
