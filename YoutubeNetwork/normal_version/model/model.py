# -*- coding: utf-8 -*-
import tensorflow as tf


class Model(object):
    def __init__(self, args):

        self.is_training = args.is_training
        self.embedding_size = args.embedding_size
        # self.basic_size=args.basic_size
        """
        brand_list保存的是所有数据去重之后，所有的brand的新的index，
        """
        self.brand_list = args.brand_list
        self.msort_list = args.msort_list
        self.item_count = args.item_count
        self.brand_count = args.brand_count
        self.msort_count = args.msort_count
        self.build_model()

    def build_model(self):
        # placeholder
        # self.u = tf.placeholder(tf.int32, [None, ])  # user idx [B]
        self.hist_click = tf.placeholder(tf.int32, [None, None])  # history click[B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # history len [B]
        self.last_click = tf.placeholder(tf.int32, [None, 1])  # last click[B]
        # self.basic = tf.placeholder(tf.float32, [None, 4])  # user basic feature[B,basic_size]
        self.sub_sample = tf.placeholder(tf.int32, [None, None])  # soft layer (pos_clict,neg_list)[B,sub_size]
        self.y = tf.placeholder(tf.float32, [None, None])  # label one hot[B]
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.lr = tf.placeholder(tf.float64, [])

        # emb variable wx+b
        item_emb_w = tf.get_variable("item_emb_w", [self.item_count, self.embedding_size])
        brand_emb_w = tf.get_variable("brand_emb_w", [self.brand_count, self.embedding_size])
        msort_emb_w = tf.get_variable("msort_emb_w", [self.msort_count, self.embedding_size])

        input_b = tf.get_variable("input_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        # 去重的所有品牌的列表
        brand_list = tf.convert_to_tensor(self.brand_list, dtype=tf.int32)
        msort_list = tf.convert_to_tensor(self.msort_list, dtype=tf.int32)

        # historty click including item brand and sort , concat as axis = 2
        """
        这里默认的brand_list应该就是我每次点击的对应的brand，然后我把这些brand根据item_id的顺序进行了一个排列，这样当我用hist_click去gather brand_list的时候，比如我第一个hist点击的物品index是0，然后在brand list中，位置0的brand的index正好是3，这样就把历史点击的index转换成为了brand的index
        """

        hist_brand = tf.gather(brand_list, self.hist_click)
        hist_sort = tf.gather(msort_list, self.hist_click)

        """
        这里不知道为啥是通过axis=2来拦截
        每个look up的维度是
        None*max_click_length*embd_size
        所有的这3个tensor，都在最后一个维度上面进行concat，最后的值就变成

        在哪个维度上面concat，就是在这个维度上面进行相加，所以最后的维度是
        None*max_click_length*(embd_size+embd_size+embd_size)
        t1 = [[1, 2, 3], [4, 5, 6]]  size=2*3
        t2 = [[7, 8, 9], [10, 11, 12]] size=2*3
        tf.concat([t1, t2], 0)  size=(2+2)*3 # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tf.concat([t1, t2], 1)  size=2*(3+3) # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

        """
        # None*max_click_length*(3*embd_size)
        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_click),
                           tf.nn.embedding_lookup(brand_emb_w, hist_brand),
                           tf.nn.embedding_lookup(msort_emb_w, hist_sort)], axis=2)

        # historty mask only calculate the click action
        """
          tf.sequence_mask([1, 3, 2], 5) =
            [[True, False, False, False, False],
             [True, True, True, False, False],
             [True, True, False, False, False]]

         shape(h_emb)[1]就是max_history_click_length

         所以这里就是把每次点击的前n个通过True和false选出来，这样在做embedding的时候，就不用多出一个维度来处理空了

        """
        mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [B,T]
        """
        增加了一个维度，为了跟上面embedding匹配
        """
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        """
        在第三个维度上面，复制3次，因为3次分别对应的是item,brand,sormidd的3个embedding
        """
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]
        """
        把之前的B,T,3*e的embedding进行一个mask
        """
        h_emb *= mask  # [B,T,3*e]

        """
        None*max_click_length*(3*embd_size)，现在click_length上面进行一个sum
        """
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        hist = tf.div(hist,
                      # 就是把每次的长度在第二个维度上面，扩展到3 * self.embedding_size次，然后用上面的
                      # embedding除以下面的这个实际的点击次数
                      tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, 3 * self.embedding_size]), tf.float32))  # [B,3*e]

        # last click including item brand and sort , concat as axis = 2
        last_b = tf.gather(brand_list, self.last_click)
        last_m = tf.gather(msort_list, self.last_click)
        last_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.last_click),
                              tf.nn.embedding_lookup(brand_emb_w, last_b),
                              tf.nn.embedding_lookup(msort_emb_w, last_m)], axis=-1)
        last_emb = tf.squeeze(last_emb, axis=1)

        # self.input = tf.concat([hist, last_emb, self.basic], axis=-1)

        # 所以这个时候的size就是 [B,6*e]
        self.input = tf.concat([hist, last_emb], axis=-1)

        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_1 = tf.nn.dropout(layer_1, keep_prob=self.keep_prob)
        layer_2 = tf.layers.dense(layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_2 = tf.nn.dropout(layer_2, keep_prob=self.keep_prob)

        # 这里最后又压缩到3*e的维度，因为我就是用brand，item,midsort这个三个embeding来弄的
        layer_3 = tf.layers.dense(layer_2, 3 * self.embedding_size, activation=tf.nn.relu, name='f3')

        # softmax
        if self.is_training:

            # find brand and sort idx
            sam_b = tf.gather(brand_list, self.sub_sample)
            sam_m = tf.gather(msort_list, self.sub_sample)

            """
            这个只是在最后w*x+b中的那个b，所以意义不是很大
            """
            sample_b = tf.nn.embedding_lookup(input_b, self.sub_sample)  # [B,sample]

            # get item/brand/sort embedding vector and concat them
            sample_w = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.sub_sample),
                                  tf.nn.embedding_lookup(brand_emb_w, sam_b),
                                  tf.nn.embedding_lookup(msort_emb_w, sam_m)
                                  # tf.tile(tf.expand_dims(self.basic, 1), [1, tf.shape(sample_b)[1], 1])
                                  ], axis=2)  # [B,sample,3*e]
            # 又增加了一个维度
            user_v = tf.expand_dims(layer_3, 1)  # [B,1,3*e]
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])  # [B,3*e,sample]

            """
            这里的矩阵相乘是[B,1,3*e] * [B,3*e,sample]
            但是其实自己在做tf思考的时候，可以假定我们的batch_size就是1，所以这里的两个相乘就变从了
            [1,3*e] * [3*e,sample]

            """
            self.logits = tf.squeeze(tf.matmul(user_v, sample_w), axis=1) + sample_b

            # Step variable
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
            self.yhat = tf.nn.softmax(self.logits)

            # 这就是一个用softmax来作为一个景点的2分类loss，我y只有第一个的label是1，其他的都是0，这里其实也正好对应了之前定义的softmax的损失函数，就是sum(yi*log(softmax(xi)))其实多余多分类，也就是我分类正确的那个i，对应的一个-的log(prob)的导数
            self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))

            trainable_params = tf.trainable_variables()
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(self.loss, trainable_params)
            """
            这个clip_global_norm的方法，其实可以学着使用
            """

            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)


        else:
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(brand_emb_w, brand_list),
                                 tf.nn.embedding_lookup(msort_emb_w, msort_list)],
                                axis=1)
            """
            如果是测试的时候，，就和all_embd进行相乘，然后用softmax确定最大的是那个
            """
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True) + input_b
            self.output = tf.nn.softmax(self.logits)

    """
    uij是一个tupple，里面包括
     (u, i, y, hist_i, sl, b, lt, qr)
    u是每个玩家的id
    i这个玩家当次点击正&负样本item的index
    y是上面那个正负样本item的index对应的label，1和0
    hist_i是吧所有玩家点击的item index对应到一个column长度均等的一个方阵,但是这里item的index对应是0的然后这里hist_i填充的也是0，所以在embedding的时候可能会出错

    """
    def train(self, sess, uij, l, keep_prob):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.sub_sample: uij[1],
            self.y: uij[2],
            # 历史点击的item的matrix，每一行代表过去点击过的物品的index
            self.hist_click: uij[3],
            # 当次点击的次数
            self.sl: uij[4],
            #  上一次点击的item_id
            self.last_click: uij[6],
            # learning rate
            self.lr: l,
            self.keep_prob: keep_prob
        })
        return loss

    def test(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            # self.basic: uij[3],
            self.hist_click: uij[1],
            self.sl: uij[2],
            self.last_click: uij[4],
            self.keep_prob: keep_prob
        })

    def eval(self, sess, uij, keep_prob):
        return sess.run(self.output, feed_dict={
            # self.basic: uij[3],
            self.hist_click: uij[1],
            self.sl: uij[2],
            self.last_click: uij[4],
            self.keep_prob: keep_prob
        })

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
