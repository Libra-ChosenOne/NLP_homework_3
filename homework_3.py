import numpy as np
import time
import jieba
import os
import random
import shutil
import os
import re


library = "./data"
out_dir = "cut"
file_ready = "cut.txt"

text_fileL = ["射雕英雄传","天龙八部","雪山飞狐","倚天屠龙记","神雕侠侣"]


# 首先去除文本的符号，只保留文字部分
re_preprocess = re.compile('[a-zA-Z0-9’"#$%&\'()*+,-./:：;<=>?@?★、…【】《》？“”‘’！[\\]^_`{|}~]+')


# 为了将文本筛选出段落，我们需要对文件进行预处理
def getCorpus(text_raw):
    text_raw = re_preprocess.sub("", text_raw)
    punctuationL = ["\t", "\n", "\u3000", "\u0020", "\u00A0", " "]
    for i in punctuationL:
        text_raw = text_raw.replace(i, "")
    text_raw = text_raw.replace("，", "。")
    corpus = text_raw.split("。")
    return corpus


if __name__ == '__main__':
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(file_ready):
        os.remove(file_ready)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for text_file in text_fileL:
        # text_file = text_fileL[0]
        with open(f"{library}/{text_file}.txt", "r", encoding="GB18030") as fp:
            text_raw = "".join(fp.readlines())
        corpus = getCorpus(text_raw)

        paraL = []
        para_len = 0
        file_id = 0
        for corpu in corpus:
            paraL.append(corpu)
            para_len += len(corpu)
            if para_len > 2000:
                para_len = 0
                with open(f"{out_dir}/{text_file}-{file_id:03d}.txt", "w", encoding="utf-8") as fp:
                    fp.writelines(paraL)
                paraL = []
                file_id += 1

        random_paramL = [i for i in range(file_id)]
        random.shuffle(random_paramL)
        random_paramL_40 = random_paramL[:40]
        random_paramL_40.sort()
        random_paramL_40 = [f"{out_dir}/{text_file}-{i:03d}.txt\n" for i in random_paramL_40]
        with open(file_ready, "a", encoding="utf-8") as fp:
            fp.writelines(random_paramL_40)
library = "./data"
file_ready = "cut.txt"

a = 6
b = 0.2
epoch_times = 100

topics = 10
start_time = time.time()

with open("stop_word.txt", 'r', encoding='utf-8') as fp:
    stopWordL = fp.readlines()
stopWordL = [i.strip() for i in stopWordL]


class LDA:
    def __init__(self) -> None:
        self.docs = None
        self.word2id_dict = None
        self.id2word_dict = None

        self.num_doc = 0
        self.num_word = 0
        self.Z = []

        # 在param_init里面初始化参数
        self.ndz = None
        self.nzw = None
        self.nz = None
        self.theta = None
        self.phi = None

    def gen_dict(self, documentL):
        word2id_dict = {}
        id2word_dict = {}
        docs = []
        cnt_document = []
        cnt_word_id = 0

        for document in documentL:
            segList = jieba.cut(document)
            for word in segList:
                word = word.strip()
                if len(word) > 1 and word not in stopWordL:
                    if word in word2id_dict:
                        cnt_document.append(word2id_dict[word])
                    else:
                        cnt_document.append(cnt_word_id)
                        word2id_dict[word] = cnt_word_id
                        id2word_dict[cnt_word_id] = word
                        cnt_word_id += 1
            docs.append(cnt_document)
            cnt_document = []
        self.docs, self.word2id_dict, self.id2word_dict = docs, word2id_dict, id2word_dict
        self.num_doc = len(self.docs)
        self.num_word = len(self.word2id_dict)

    # 初始化
    def param_init(self):
        self.ndz = np.zeros([self.num_doc, topics]) + a
        self.nzw = np.zeros([topics, self.num_word]) + b
        self.nz = np.zeros([topics]) + self.num_word * b
        self.theta = np.zeros([self.num_doc, topics])
        self.phi = np.zeros([topics, self.num_word])

        for d, doc in enumerate(self.docs):
            zCurrentDoc = []
            for w in doc:
                self.pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z = np.random.multinomial(1, self.pz / self.pz.sum()).argmax()
                zCurrentDoc.append(z)
                self.ndz[d, z] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
            self.Z.append(zCurrentDoc)

    # 采样
    def gibbs_sampling_update(self):
        for d, doc in enumerate(self.docs):
            for index, w in enumerate(doc):
                z = self.Z[d][index]
                self.ndz[d, z] -= 1
                self.nzw[z, w] -= 1
                self.nz[z] -= 1
                self.pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
                z = np.random.multinomial(1, self.pz / self.pz.sum()).argmax()
                self.Z[d][index] = z
                self.ndz[d, z] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1

        self.theta = [(self.ndz[i] + a) / (len(self.docs[i]) + topics * a) for i in range(self.num_doc)]
        self.phi = [(self.nzw[i] + b) / (self.nz[i] + self.num_word * b) for i in range(topics)]

    def cal_Complexity(self):
        nd = np.sum(self.ndz, 1)
        n = 0
        ll = 0.0
        for d, doc in enumerate(self.docs):
            for w in doc:
                ll = ll + np.log(((self.nzw[:, w] / self.nz) * (self.ndz[d, :] / nd[d])).sum())
                n = n + 1
        return np.exp(ll / (-n))

    def show_topwords(self, num=10):
        for z in range(topics):
            ids = self.nzw[z, :].argsort()
            topicword = []
            for j in ids:
                topicword.insert(0, self.id2word_dict[j])
            # topicwords.append(topicword[:min(num, len(topicword))])
            print(topicword[:min(num, len(topicword))])

    def save_param(self, postfix):
        np.savetxt(f"theta_{postfix:03d}.csv", self.theta, fmt="%.9f", delimiter=',')
        np.savetxt(f"phi_{postfix:03d}.csv", self.phi, fmt="%.9f", delimiter=',')


if __name__ == '__main__':
    hw_lda = LDA()

    documentL = []
    with open(file_ready, "r", encoding="utf-8") as fp:
        fileL = fp.readlines()
    for file in fileL:
        with open(file.strip(), 'r', encoding='utf-8') as f:
            documentL.append(f.read())

    hw_lda.gen_dict(documentL)
    print("gen_dict done")

    hw_lda.param_init()

    ComplexityL = []
    for i in range(epoch_times):
        hw_lda.gibbs_sampling_update()
        Complexity = hw_lda.cal_Complexity()
        ComplexityL.append(Complexity)
        if not i % 10:
            hw_lda.save_param(i)

    np.savetxt("Complexity.csv", ComplexityL, fmt="%.9f", delimiter=',')

    hw_lda.show_topwords()