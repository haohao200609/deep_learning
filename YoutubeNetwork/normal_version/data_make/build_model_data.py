# -*- coding: utf-8 -*-
import random
import pickle
import numpy as np

random.seed(1234)

with open('remap.pkl', 'rb') as f:
    userclick_data = pickle.load(f)
    item_key, brand_key, msort_key, user_key = pickle.load(f)
    brand_list = pickle.load(f)
    msort_list = pickle.load(f)
    user_count, item_count, brand_count,msort_count, example_count = pickle.load(f)

    print('user_count: %d\titem_count: %d\tbrand_count: %d\texample_count: %d' %
      (user_count, item_count, brand_count, example_count))
    train_set = []
    test_set = []
    uid_num=0
    """

    df
       Animal  Max Speed
    0  Falcon      380.0
    1  Falcon      370.0
    2  Parrot       24.0
    3  Parrot       26.0

    for name,value_list in df.groupby(['Animal']):
        print name
        print 'dd'
        print type(value_list)

    Falcon
    dd
    <class 'pandas.core.frame.DataFrame'>
    Parrot
    dd
    <class 'pandas.core.frame.DataFrame'>


    for name,value_list in df.groupby(['Animal']):
        print name
        print 'dd'
        print value_list

    Falcon
    dd
       Animal  Max Speed
    0  Falcon      380.0
    1  Falcon      370.0
    Parrot
    dd
       Animal  Max Speed
    2  Parrot       24.0
    3  Parrot       26.0

    """


    """
    我这里只能假设这里的数据是按照时间顺序一条条存储的

    """
    for UId, hist in userclick_data.groupby('UId'):
        uid_num+=1
        #print('uid_num',uid_num)
        #print(hist)
        """
        pos_list应该是玩家实际点击过的所有item_id
        这里是没有一个时间顺序的，如果itemid按照时间排序可能会好一些
        就是按照从上到下进行排序，而且tolist()没有去重

        """
        pos_list = hist['ItemId'].tolist()
        if len(pos_list)<3:
            print('one')
            continue
        #print(pos_list)
        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg
        """
        这里就是产生了正样本20倍的负样本的item index
        """
        neg_list = [gen_neg() for i in range(20*len(pos_list))]
        #print(neg_list)
        neg_list=np.array(neg_list)
        #print(neg_list)
        #print('len pos',len(pos_list))
        for i in range(1, len(pos_list)):
            index = np.random.randint(len(neg_list), size=20)
            #print(index)
            """
            这里也就是记录，在当前这个物品之前，我点击过哪些物品
            """
            hist = pos_list[:i]
            #print('i',i)
            if i!= len(pos_list) :
                #print('if')
                #print(neg_list[index])
                """
                这一条记录就是，
                用户id，我历史点击了那些item对应的index，当前点击的这个item index，我neg则是我用户完全没有点击或的item index（这里的负样本是从整体取的，也就是我整个train set中，我用户没有点击的item_id，我可能第1天点击了item0，第二天点击了item2,所以在制造item0的负样本的时候，是没有item2的）
                """
                train_set.append((UId, hist, pos_list[i], list(neg_list[index])))
                #train_set.append((UId, hist, neg_list[i], 0))
            #else:
                #print('test',uid_num)
                #label = (pos_list[i], neg_list[i])
                #test_set.append((UId, hist,label))
            #break

        """
        这里测试的时候，如果我某个用户点击的数量大于20，我就去最后的20个作为我的测试集，如果我用户不到20个，就去所有的pos_list来作为测试集

        但是这里测试集和训练集，其实用的数据是同一份
        """
        if len(pos_list)>20:
            test_set.append((UId, pos_list[-20:]))
        else:
            test_set.append((UId, pos_list))

        #break
print(len(train_set))
train_set_1=train_set[:400000]
train_set_2=train_set[400000:800000]
train_set_3=train_set[800000:]
# print(train_set[:12])
random.shuffle(train_set)
random.shuffle(test_set)
#print(len(train_set))

#assert len(test_set) == user_count
print('test len',len(test_set))
print('user count',user_count)
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
    print('train')
    pickle.dump(train_set_1, f, pickle.HIGHEST_PROTOCOL)
    print('2')
    pickle.dump(train_set_2, f, pickle.HIGHEST_PROTOCOL)
    print('3')
    pickle.dump(train_set_3, f, pickle.HIGHEST_PROTOCOL)
    print('test')
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, brand_count,msort_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_key, brand_key, msort_key,user_key) , f, pickle.HIGHEST_PROTOCOL)
