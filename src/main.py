import os
import sys
import pickle
import preprocessing_git
from gensim.models import Word2Vec, KeyedVectors
from word2vecUtils import process_embedding
from train import train_model, fill_csv

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100) + 1
    r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num, )
    sys.stdout.write(r)
    sys.stdout.flush()


def prepareDataset():
    data = []
    dir_file = '../tweet_data'
    files = os.listdir(dir_file)

    for i in range(len(files)):
        with open(os.path.join(dir_file, files[i])) as f:
            lines = f.readlines()
            idx = 0
            for line in lines:
                view_bar(idx, len(lines))
                data.append(preprocessing_git.create_dataset_word2Vec(line))
                idx += 1
            print('\n')

    with open('data.bin', 'wb') as fp:
       pickle.dump(data, fp)

def smallDataset():
    dir_file = '../tweet_data'
    data = []
    with open(os.path.join(dir_file, 'second_debate_presidential.txt')) as f:
        lines = f.readlines()
        idx = 0
        for line in lines:
            view_bar(idx, len(lines))
            data.append(preprocessing_git.create_dataset_word2Vec(line))
            idx += 1
        print('\n')
    with open('data2.bin', 'wb') as fp:
       pickle.dump(data, fp)


def createWord2Vec(modelFile, dataset):
    data = pickle.load(open(dataset, 'rb'))
    
    # train model
    model = Word2Vec(data, size=300, window=3, min_count=5, sg=1, workers=8)

    # save model
    model.save(modelFile)
    
def process_word2Vec(modelFile, words):
    model = Word2Vec.load(modelFile)
    for word in words:
        print(word)
        print(model.most_similar(word))    


if __name__ == '__main__':
    process_embedding()
    tokenizer, max_len = train_model()

    # uncomment and change csvpath to fill a file with prediction from our model
    # since train_model save our best model in './model7.h5'we use this path to get it

    # fill_csv(csvpath, './model7.h5', tokenizer, max_len)
