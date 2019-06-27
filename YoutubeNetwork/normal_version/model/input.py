# -*- coding: utf-8 -*-
import numpy as np

"""

这一条记录就是，
用户id，我历史点击了那些item对应的index，当前点击的这个item index，我neg则是我用户完全没有点击或的item index（这里的负样本是从整体取的，也就是我整个train set中，我用户没有点击的item_id，我可能第1天点击了item0，第二天点击了item2,所以在制造item0的负样本的时候，是没有item2的）

train_set.append((UId, hist, pos_list[i], list(neg_list[index])))
所以这里的data就是上面的这个列表
"""

class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1
        """
        train_set.append((UId, hist, pos_list[i], list(neg_list[index])))
        """
        u, i, y, sl, b, lt, qr = [], [], [], [], [], [], []
        for t in ts:
            #用户id
            u.append(t[0])
            # i是正样本+负样本 1：20
            i.append([t[2]] + t[3])
            # 样本数
            sub_sample_size = len(t[3]) + 1
            mask = np.zeros(sub_sample_size, np.int64)
            # 用来标记i里面第一个是正样本，剩下的都是0
            mask[0] = 1
            y.append(mask)
            # 历史点击item的数量
            sl.append(len(t[1]))  # histroy click seq
            # 这个似乎没用到
            b.append(t[4])
            # 这个也没用到
            lt.append(t[5])
            # qr.append(t[6])
        # 获得一个最长的历史点击物品的概率
        max_sl = max(sl)
        # hist_i应该是把之前点击的长度不一的list嵌套list，转为一个colum长度相等的hist_i,如果长度不够max_sl的最后就是0
        hist_i = np.zeros([len(ts), max_sl], np.int64)
        # print('u',u)
        # print('i',len(i))
        # print('y',y)
        # print('hist_i',hist_i)
        # print('sl',sl)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        # print('hist_i',hist_i)
        """
        u是每个玩家的id
        i这个玩家当次点击正负样本item的index
        y是上面那个正负样本item的index对应的label，1和0
        hist_i是吧所有玩家点击的item index对应到一个column长度均等的一个方阵,但是这里item的index对应是0的然后这里hist_i填充的也是0，所以在embedding的时候可能会出错

        """
        return self.i, (u, i, y, hist_i, sl, b, lt, qr)


class DataInputTest:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, sl, b, lt, qr = [], [], [], [], []
        for t in ts:
            u.append(t[0])
            sl.append(len(t[1]))  # histroy click seq
            b.append(t[2])
            lt.append(t[3])
            # qr.append(t[4])

        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        # print('hist_i',hist_i)

        return self.i, (u, hist_i, sl, b, lt, qr)


class DataInputEval:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        u, sl, b, lt, now, future = [], [], [], [], [], []
        for t in ts:
            u.append(t[0])
            sl.append(len(t[1]))  # histroy click seq
            # b.append(t[4])
            lt.append(t[1][-1:])
            now.append(t[2])
            future.append(t[3])
            # qr.append(t[4])

        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        # print('hist_i',hist_i)

        return self.i, (u, hist_i, sl, b, lt, now,future)
