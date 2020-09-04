import tensorflow as tf
import numpy as np
import os
import json
import sys
sys.path.append("..")
from data.DanceDataset import DanceDataset
from MotionVae import MotionVae
from MusicVae import MusicVae
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.framework.python.ops import add_arg_scope
class VAE_LSTM_FIX_TCV_model:
    def __init__(self,
                 train_file_list,
                 model_save_dir,
                 model_load_dir,
                 log_dir,
                 motion_vae_ckpt_dir,
                 music_vae_ckpt_dir,
                 rnn_unit_size=32,
                 acoustic_dim=16,
                 temporal_dim=3,
                 motion_dim=63,
                 time_step=120,
                 batch_size=10,
                 learning_rate=1e-3,
                 extr_loss_threshold=0.045,
                 overlap=True,
                 epoch_size=1500,
                 use_mask=True,
                 normalize_mode='minmax'
                 ):
        # lstm
        self.model_save_dir=model_save_dir
        self.model_load_dir=model_load_dir
        self.log_dir=log_dir
        self.motion_vae_ckpt_dir=motion_vae_ckpt_dir
        self.music_vae_ckpt_dir=music_vae_ckpt_dir
        self.dense_dim=16
        self.rnn_input_dim = 32
        self.rnn_output_dim = 32
        self.rnn_unit_size = rnn_unit_size
        self.acoustic_dim = acoustic_dim
        self.temporal_dim = temporal_dim
        self.motion_dim = motion_dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.extr_loss_threshold = extr_loss_threshold
        self.train_file_list = train_file_list
        self.overlap = overlap
        self.epoch_size = epoch_size
        self.use_mask = use_mask

        #vae
        self.n_hidden = 500
        self.music_latent_dim=8
        self.motion_latent_dim=16

        self.train_dataset = DanceDataset(train_file_list=self.train_file_list,
                                          acoustic_dim=self.acoustic_dim,
                                          temporal_dim=self.temporal_dim,
                                          motion_dim=self.motion_dim,
                                          time_step=self.time_step,
                                          overlap=self.overlap,
                                          overlap_interval=10,
                                          batch_size=self.batch_size,
                                          normalize_mode=normalize_mode)


        self.musicVae=MusicVae()
        self.motionVae=MotionVae()

    def lstm_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.rnn_unit_size)

    def get_name(self,layer_name, counters):
        ''' utlity for keeping track of layer names '''
        if not layer_name in counters:
            counters[layer_name] = 0
        name = layer_name + '_' + str(counters[layer_name])
        counters[layer_name] += 1
        return name

    def temporal_padding(self,x, padding=(1, 1)):
        """Pads the middle dimension of a 3D tensor.
        # Arguments
            x: Tensor or variable.
            padding: Tuple of 2 integers, how many zeros to
                add at the start and end of dim 1.
        # Returns
            A padded 3D tensor.
        """
        assert len(padding) == 2
        pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
        return tf.pad(x, pattern)

    @add_arg_scope
    def weightNormConvolution1d(self,x, num_filters, dilation_rate, filter_size=3, stride=[1],
                                pad='VALID', init_scale=1., init=False, gated=False,
                                counters={}, reuse=False):
        """a dilated convolution with weight normalization (Salimans & Kingma 2016)
           Note that init part is NEVER used in our code
           It relates to the data-dependent init in original paper
        # Arguments
            x: A tensor of shape [N, L, Cin]
            num_filters: number of convolution filters
            dilation_rate: dilation rate / holes
            filter_size: window / kernel width of each filter
            stride: stride in convolution
            gated: use gated linear units (Dauphin 2016) as activation
        # Returns
            A tensor of shape [N, L, num_filters]
        """
        name = self.get_name('weight_norm_conv1d', counters)
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # currently this part is never used
            if init:
                print("initializing weight norm")
                # data based initialization of parameters
                V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                    tf.float32, tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
                V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])

                # pad x
                left_pad = dilation_rate * (filter_size - 1)
                x = self.temporal_padding(x, (left_pad, 0))
                x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
                # x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
                m_init, v_init = tf.nn.moments(x_init, [0, 1])
                scale_init = init_scale / tf.sqrt(v_init + 1e-8)
                g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init,
                                    trainable=True)
                b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init * scale_init,
                                    trainable=True)
                x_init = tf.reshape(scale_init, [1, 1, num_filters]) \
                         * (x_init - tf.reshape(m_init, [1, 1, num_filters]))
                # apply nonlinearity
                x_init = tf.nn.relu(x_init)
                return x_init

            else:
                # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
                if gated:
                    num_filters = num_filters * 2

                # size of V is L, Cin, Cout
                V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                    tf.float32, tf.random_normal_initializer(0, 0.01),
                                    trainable=True)
                g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.), trainable=True)
                b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                    initializer=None, trainable=True)

                # size of input x is N, L, Cin

                # use weight normalization (Salimans & Kingma, 2016)
                W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])

                # pad x for causal convolution
                left_pad = dilation_rate * (filter_size - 1)
                x = self.temporal_padding(x, (left_pad, 0))

                # calculate convolutional layer output
                x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)

                # GLU
                if gated:
                    split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                    split1 = tf.sigmoid(split1)
                    x = tf.multiply(split0, split1)
                # ReLU
                else:
                    # apply nonlinearity
                    x = tf.nn.relu(x)

                print(x.get_shape())

                return x

    def TemporalBlock(self,input_layer, out_channels, filter_size, stride, dilation_rate, counters,
                      dropout, init=False, atten=False, use_highway=False, gated=False):
        """temporal block in TCN (Bai 2018)
        # Arguments
            input_layer: A tensor of shape [N, L, Cin]
            out_channels: output dimension
            filter_size: receptive field of a conv. filter
            stride: same as what's need in conv. function
            dilation_rate: holes inbetween
            counters: to keep track of layer names
            dropout: prob. to drop weights
            atten: (not in TCN) add self attention block after Conv.
            use_highway: (not in TCN) use highway as residual connection
            gated: (not in TCN) use gated linear unit as activation
            init: (NEVER used) data-dependent initialization
        # Returns
            A tensor of shape [N, L, out_channels]
        """
        keep_prob = 1.0 - dropout

        in_channels = input_layer.get_shape()[-1]
        name = self.get_name('temporal_block', counters)
        with tf.variable_scope(name):

            # num_filters is the hidden units in TCN
            # which is the number of out channels
            conv1 = self.weightNormConvolution1d(input_layer, out_channels, dilation_rate,
                                            filter_size, [stride], counters=counters,
                                            init=init, gated=gated)
            # set noise shape for spatial dropout
            # refer to https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
            # shape should be [N, 1, C]
            noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
            out1 = tf.nn.dropout(conv1, keep_prob, noise_shape)
            if atten:
                out1 = self.attentionBlock(out1, counters, dropout)

            conv2 = self.weightNormConvolution1d(out1, out_channels, dilation_rate, filter_size,
                                            [stride], counters=counters, init=init, gated=gated)
            out2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
            if atten:
                out2 =self.attentionBlock(out2, counters, dropout)

            # highway connetions or residual connection
            residual = None
            if use_highway:
                W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                      tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
                b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                      initializer=None, trainable=True)
                H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

                W_t = tf.get_variable('W_t', [1, int(input_layer.get_shape()[-1]), out_channels],
                                      tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
                b_t = tf.get_variable('b_t', shape=[out_channels], dtype=tf.float32,
                                      initializer=None, trainable=True)
                T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
                T = tf.nn.sigmoid(T)
                residual = H * T + input_layer * (1.0 - T)
            elif in_channels != out_channels:
                W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                      tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
                b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                      initializer=None, trainable=True)
                residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
            else:
                print("no residual convolution")

            res = input_layer if residual is None else residual

            return tf.nn.relu(out2 + res)

    def acoustic_features_extractor(self,acoustic_input,acoustic_target,temporal_input,mask_input,trainable):
        batch_size = tf.shape(acoustic_input)[0]
        attn_cell = self.lstm_cell
        counters = {}
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )
        # ----------------------------------music encoder-------------------------------------
        acoustic_input=tf.reshape(acoustic_input, [-1,self.acoustic_dim]) 
        acoustic_mean, acoustic_stddev=self.musicVae.music_encoder(acoustic_input,self.n_hidden,self.music_latent_dim)# bacthsize*timestep, 8

        acoustic_latent= acoustic_mean + acoustic_stddev * tf.random_normal(tf.shape(acoustic_mean), 0, 1, dtype=tf.float32)

        acoustic_latent=tf.reshape(acoustic_latent, [-1,self.time_step,self.music_latent_dim])
        # ----------------------------------dense 1-------------------------------------
        # with tf.variable_scope("dense_1"):
        #     dense_1 = tf.layers.dense(acoustic_latent,
        #                               self.dense_dim,
        #                               activation=tf.nn.relu,
        #                               trainable=trainable)
        dense_1=acoustic_latent
        num_channels=[36,24,16]
        num_levels = len(num_channels)
        for i in range(num_levels):
            print(i)
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            dense_1 = self.TemporalBlock(dense_1, out_channels, filter_size=2, stride=1,
                                         dilation_rate=dilation_size,
                                         counters=counters, dropout=0.0, init=False, atten=False, gated=False)


        # ----------------------------------lstm 2-------------------------------------
        with tf.variable_scope("lstm_2"):

            concat2 = tf.concat([dense_1, temporal_input], 2)
            concat2 = tf.reshape(concat2, [-1,self.dense_dim + self.temporal_dim])
            concat_rnn2 = tf.nn.bias_add(tf.matmul(concat2,tf.Variable(tf.truncated_normal([self.dense_dim + self.temporal_dim, self.rnn_input_dim]))),
                                         bias=tf.Variable(tf.zeros(shape=[self.rnn_input_dim])))
            concat_rnn2 = tf.reshape(concat_rnn2, [-1, self.time_step, self.rnn_input_dim])
            cell2 = tf.contrib.rnn.MultiRNNCell( [attn_cell() for _ in range(3)] )
            init_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
            output_rnn2, final_states2 = tf.nn.dynamic_rnn(cell2, concat_rnn2,
                                                           initial_state=init_state2,
                                                           dtype=tf.float32)
            output2 = tf.reshape(output_rnn2, [-1, self.rnn_unit_size])
            pred2 = tf.nn.bias_add(tf.matmul(output2, tf.Variable(tf.truncated_normal([self.rnn_unit_size, self.rnn_output_dim]))),
                                   bias=tf.Variable(tf.zeros(shape=[self.rnn_output_dim])))
            pred2 = tf.reshape(pred2, [-1, self.time_step, self.rnn_output_dim])
        # ----------------------------------mask 3-------------------------------------
        with tf.variable_scope("mask_4"):
            if self.use_mask:
                reduced_acoustic_features = pred2
            else:
                mask = mask_input
                reduced_acoustic_features = tf.multiply(pred2, mask)    
        # ----------------------------------dense 4-------------------------------------
        with tf.variable_scope("dense_5"):
            motion_latent = tf.layers.dense(reduced_acoustic_features,
                                     self.motion_latent_dim,
                                     activation=None,
                                     trainable=trainable)
            motion_latent=tf.reshape(motion_latent,[-1,self.motion_latent_dim])

        # ----------------------------------music decoder-------------------------------------

        acoustic_latent=tf.reshape(acoustic_latent, [-1,self.music_latent_dim])


        music_predict=self.musicVae.music_decoder(acoustic_latent,  self.n_hidden, self.acoustic_dim)
        music_predict=tf.clip_by_value(music_predict, 1e-8, 1 - 1e-8)
        acoustic_target=tf.reshape(acoustic_target,[-1,self.acoustic_dim])
        music_loss = tf.losses.mean_squared_error(acoustic_target ,music_predict)


        return motion_latent,music_loss

    def motion_predictor(self,motion_input,motion_latent,temporal_input,trainable):
        batch_size = tf.shape(motion_input)[0]
        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell(),
                    output_keep_prob=0.9
                )
        # ----------------------------------motion encoder-------------------------------------
        motion_input=tf.reshape(motion_input, [-1,self.motion_dim]) 
        motion_mean, motion_stddev=self.motionVae.motion_encoder(motion_input,self.n_hidden,self.motion_latent_dim)# bacthsize*timestep, 8

        motion_latent_gt= motion_mean + motion_stddev * tf.random_normal(tf.shape(motion_mean), 0, 1, dtype=tf.float32)

        loss_motion_latent = tf.losses.mean_squared_error(motion_latent_gt, motion_latent)

        
        # ----------------------------------motion decoder-------------------------------------
        motion_predict=self.motionVae.motion_decoder(motion_latent,  self.n_hidden, self.motion_dim)
        motion_predict = tf.clip_by_value(motion_predict, 1e-8, 1 - 1e-8)
        motion_predict=tf.reshape(motion_predict,[-1,self.time_step,self.motion_dim])


        return motion_predict,loss_motion_latent

    def motion_val(self,motion_latent, temporal_input, trainable):
        batch_size = tf.shape(temporal_input)[0]
        attn_cell = self.lstm_cell
        if trainable:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    self.lstm_cell()
                )
        # ----------------------------------motion decoder-------------------------------------
        motion_predict = self.motionVae.motion_decoder(motion_latent, self.n_hidden, self.motion_dim)
        motion_predict = tf.clip_by_value(motion_predict, 1e-8, 1 - 1e-8)
        motion_predict = tf.reshape(motion_predict, [-1, self.time_step, self.motion_dim])

        return motion_predict

    def train(self,resume=False):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])

        motion_latent,music_loss = self.acoustic_features_extractor(acoustic,acoustic,temporal,mask,trainable=True)
        predicted_motion_features,loss_motion_latent = self.motion_predictor(motion,motion_latent,temporal,trainable=True)
        # loss
        loss_extr = music_loss
        motion_loss=tf.losses.mean_squared_error(predicted_motion_features, motion)
        loss_pred = motion_loss+loss_motion_latent
        loss = tf.maximum(self.extr_loss_threshold, loss_extr) + loss_pred
        tf.summary.scalar("motion_loss", motion_loss)
        tf.summary.scalar("music_loss", music_loss)
        tf.summary.scalar("loss_motion_latent", loss_motion_latent)
        tf.summary.scalar("loss_pred", loss_pred)
        tf.summary.scalar("loss", loss)
        

        train_var_list = [var for var in tf.trainable_variables() if ('decoder' not in var.name and 'encoder' not in var.name  )] 
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999).minimize(loss, var_list = train_var_list) #自行选择优化器
        
        saver = tf.train.Saver(train_var_list, max_to_keep=5)

        iterator = self.train_dataset.train_dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        step_num=self.train_dataset.train_size // self.batch_size
        print("step_size: %d" % (step_num))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # # restore pretrained weights
            print('loading weight from %s...'%self.motion_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.motion_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'motion' in vv.name):
                    print("load %s"%vv.name[:-2])
                    weights = reader.get_tensor(vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
            print('loading weight from %s...' % self.music_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.music_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'music' in vv.name):
                    print("load %s" % 'music_autoencoder/'+vv.name[:-2])
                    weights = reader.get_tensor('music_autoencoder/'+vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)    
            
            if resume:
                ckpt = tf.train.get_checkpoint_state(self.model_load_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("restore weight from %s ..."%self.model_load_dir)
                    saver.restore(sess, ckpt.model_checkpoint_path)
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            writer.add_graph(sess.graph)
            summ = tf.summary.merge_all()
            for i in range(self.epoch_size):
                print("epoch:%d" % i)
                loss_avg = 0
                sess.run(iterator.initializer)
                for step in range(step_num):
                    batch_data = sess.run(next_element)
                    acoustic_in = batch_data[:, :, :self.acoustic_dim]
                    temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]
                    motion_in = batch_data[:, :,
                                self.acoustic_dim + self.temporal_dim:self.acoustic_dim + self.temporal_dim + self.motion_dim]
                    mask_in = temporal_in[:, :, 1]
                    mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                    _, loss_,motion_loss_, loss_e,loss_m,loss_ml, loss_p, sum = sess.run([train_op, loss,motion_loss, loss_extr,music_loss,loss_motion_latent, loss_pred, summ],
                                                             feed_dict={
                                                                        acoustic: acoustic_in,
                                                                        temporal: temporal_in,
                                                                        motion: motion_in,
                                                                        mask: mask_in
                                                             })
                    loss_avg += loss_
                    if step % 10 == 0:
                        print("epoch: %d step: %d, total loss: %.9f,motion_loss: %.9f, extr loss: %.9f,music_loss: %.9f,loss_motion_latent: %.9f, predict loss: %.9f " % (
                            i, step, loss_,motion_loss_, loss_e, loss_m,loss_ml,loss_p))
                writer.add_summary(sum, i)
                print("epoch: %d loss_avg: %f, " % (i, loss_avg / step))
                if (i + 1) % 10 == 0:
                    print("保存模型：", saver.save(sess,os.path.join(self.model_save_dir,'stock2.model'), global_step=i))

            writer.close()

    def predict(self,test_file,result_save_dir):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        motion = tf.placeholder(tf.float32, shape=[None, self.time_step, self.motion_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])


        motion_latent, music_loss = self.acoustic_features_extractor(acoustic, acoustic, temporal, mask, trainable=True)
        predicted_motion_features, loss_motion_latent = self.motion_predictor(motion, motion_latent, temporal,
                                                                              trainable=True)

        loss_pred = tf.losses.mean_squared_error(predicted_motion_features, motion)
        train_var_list = [var for var in tf.trainable_variables() if
                          ('decoder' not in var.name and 'encoder' not in var.name)]
        saver = tf.train.Saver(train_var_list)
        with tf.Session() as sess:
            print('loading weight from %s...' % self.motion_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.motion_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'motion' in vv.name):
                    print("load %s" % vv.name[:-2])
                    weights = reader.get_tensor(vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
            print('loading weight from %s...' % self.music_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.music_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'music' in vv.name):
                    print("load %s" % 'music_autoencoder/' + vv.name[:-2])
                    weights = reader.get_tensor('music_autoencoder/' + vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
            module_file = tf.train.latest_checkpoint(self.model_load_dir)
            saver.restore(sess, module_file)
            file_name = os.path.basename(test_file) + '.json'
            print("test the file %s" % file_name)
            loss = 0
            result_test_predict=[]
            result_motion_test = []
            for start in range(self.time_step//10-1):
                test_dataset, train_motion_scaler, test_size, center = self.train_dataset.load_test_data(test_file,start=start*10)

                if (test_size % self.batch_size == 0):
                    test_size = test_size // self.batch_size
                else:
                    test_size = test_size // self.batch_size + 1

                iterator = test_dataset.make_initializable_iterator()
                next_element = iterator.get_next()
                test_predict = np.zeros([start*10,self.motion_dim])
                motion_test = np.zeros([start*10,self.motion_dim])
                sess.run(iterator.initializer)
                for step in range(test_size):
                    batch_data = sess.run(next_element)
                    batch_data.reshape([-1, self.time_step,self.acoustic_dim + self.temporal_dim + self.motion_dim])
                    acoustic_in = batch_data[:, :, :self.acoustic_dim]
                    temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]
                    motion_in = batch_data[:, :,
                                self.acoustic_dim + self.temporal_dim:self.acoustic_dim + self.temporal_dim + self.motion_dim]
                    mask_in = temporal_in[:, :, 1]
                    mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                    prob, loss_ = sess.run([predicted_motion_features, loss_pred], feed_dict={acoustic: acoustic_in,
                                                                                              temporal: temporal_in,
                                                                                              motion: motion_in,
                                                                                              mask: mask_in})
                    predict = prob.reshape((-1,  self.motion_dim))
                    motion_in = motion_in.reshape((-1,  self.motion_dim))
                    test_predict = np.append(test_predict, predict, axis=0)
                    motion_test = np.append(motion_test, motion_in, axis=0)
                    loss += loss_
                result_test_predict.append(test_predict)
                result_motion_test.append(motion_test)

            min_length=999999
            for i in range(len(result_test_predict)):
                min_length=min(min_length,result_test_predict[i].shape[0])

            final_res=result_test_predict[0][self.time_step:min_length,:]
            final_gt=result_motion_test[0][self.time_step:min_length,:]
            for i in range(1,len(result_test_predict)):
                final_res+=result_test_predict[i][self.time_step:min_length,:]
                final_gt+=result_motion_test[i][self.time_step:min_length,:]
            final_res/=len(result_test_predict)
            final_gt/=len(result_test_predict)
            final_res=np.append(result_test_predict[0][:self.time_step,:],final_res,0)
            final_gt = np.append(result_motion_test[0][:self.time_step, :], final_gt, 0)

            test_predict = train_motion_scaler.inverse_transform(final_res)
            motion_test = train_motion_scaler.inverse_transform(final_gt)
            acc = np.average(np.abs(test_predict - motion_test))

            test_predict = np.reshape(test_predict, [-1, self.motion_dim//3, 3])
            length = test_predict.shape[0]
            test_predict = test_predict.tolist()
            center=center.tolist()
            data = {"length": length, "skeletons": test_predict,"center":center}
            with open(os.path.join(result_save_dir,file_name), 'w') as file_object:
                json.dump(data, file_object)
            print(loss, acc)

    def smooth(self,a, WSZ):
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[:WSZ - 1])[::2] / r
        stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
        return np.concatenate((start, out0, stop))

    def smooth_skeleton(self,motion):
        WSZ = 3
        skeletons_num = motion.shape[1]
        skeletons = np.hsplit(motion, skeletons_num)
        cur_skeleton = np.reshape(skeletons[0], (-1, 3))
        # print(cur_skeleton.shape)
        x_seq = np.split(cur_skeleton, 3, axis=1)[0]
        x_seq = np.reshape(x_seq, -1)

        y_seq = np.split(cur_skeleton, 3, axis=1)[1]
        y_seq = np.reshape(y_seq, -1)

        z_seq = np.split(cur_skeleton, 3, axis=1)[2]
        z_seq = np.reshape(z_seq, -1)

        x_smooth = self.smooth(x_seq, WSZ)
        y_smooth = self.smooth(y_seq, WSZ)
        z_smooth = self.smooth(z_seq, WSZ)
        x_smooth = np.array(x_smooth)
        smooth_result = np.column_stack((x_smooth, y_smooth, z_smooth))

        for i in range(1, motion.shape[1]):
            cur_skeleton = np.reshape(skeletons[i], (-1, 3))
            # print(cur_skeleton.shape)
            x_seq = np.split(cur_skeleton, 3, axis=1)[0]
            x_seq = np.reshape(x_seq, -1)

            y_seq = np.split(cur_skeleton, 3, axis=1)[1]
            y_seq = np.reshape(y_seq, -1)

            z_seq = np.split(cur_skeleton, 3, axis=1)[2]
            z_seq = np.reshape(z_seq, -1)

            x_smooth = self.smooth(x_seq, WSZ)
            y_smooth = self.smooth(y_seq, WSZ)
            z_smooth = self.smooth(z_seq, WSZ)
            x_smooth = np.array(x_smooth)
            # print(x_smooth.shape)
            x = np.linspace(1, 5050, 5050)  # X轴数据
            tmp = np.column_stack((x_smooth, y_smooth, z_smooth))
            if i == 1:
                smooth_result = np.stack((smooth_result, tmp), axis=1)
            else:
                tmp_ = tmp[:, np.newaxis, :]
                smooth_result = np.concatenate((smooth_result, tmp_), axis=1)

        return smooth_result

    def predict_from_music(self,acoustic_features, temporal_indexes,music_name,result_save_dir):
        acoustic = tf.placeholder(tf.float32, shape=[None, self.time_step, self.acoustic_dim])
        temporal = tf.placeholder(tf.float32, shape=[None, self.time_step, self.temporal_dim])
        mask = tf.placeholder(tf.float32, shape=[None, self.time_step, 1])


        motion_latent, music_loss = self.acoustic_features_extractor(acoustic, acoustic, temporal, mask, trainable=True)
        predicted_motion_features = self.motion_val(motion_latent, temporal,trainable=True)

        train_var_list = [var for var in tf.trainable_variables() if
                          ('decoder' not in var.name and 'encoder' not in var.name)]
        saver = tf.train.Saver(train_var_list)


        with tf.Session() as sess:
            print('loading weight from %s...' % self.motion_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.motion_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'motion' in vv.name):
                    print("load %s" % vv.name[:-2])
                    weights = reader.get_tensor(vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
            print('loading weight from %s...' % self.music_vae_ckpt_dir)
            reader = pywrap_tensorflow.NewCheckpointReader(self.music_vae_ckpt_dir)
            # 逐层遍历参数并替换
            for vv in tf.trainable_variables():
                if (('decoder' in vv.name or 'encoder' in vv.name) and 'music' in vv.name):
                    print("load %s" % 'music_autoencoder/' + vv.name[:-2])
                    weights = reader.get_tensor('music_autoencoder/' + vv.name[:-2])
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
            module_file = tf.train.latest_checkpoint(self.model_load_dir)
            saver.restore(sess, module_file)
            file_name = os.path.basename(music_name) + '.json'
            print("test the file %s" % file_name)
            loss = 0
            result_test_predict=[]
            result_motion_test = []
            for start in range(self.time_step//10-1):
                test_dataset, train_motion_scaler, test_size, center = self.train_dataset.generate_test_data(acoustic_features, temporal_indexes,start=start*10)

                if (test_size % self.batch_size == 0):
                    test_size = test_size // self.batch_size
                else:
                    test_size = test_size // self.batch_size + 1

                iterator = test_dataset.make_initializable_iterator()
                next_element = iterator.get_next()
                test_predict = np.zeros([start*10,self.motion_dim])
                sess.run(iterator.initializer)
                for step in range(test_size):
                    batch_data = sess.run(next_element)
                    batch_data.reshape([-1, self.time_step,self.acoustic_dim + self.temporal_dim ])
                    acoustic_in = batch_data[:, :, :self.acoustic_dim]
                    temporal_in = batch_data[:, :, self.acoustic_dim:self.acoustic_dim + self.temporal_dim]

                    mask_in = temporal_in[:, :, 1]
                    mask_in = np.reshape(mask_in, [-1, self.time_step, 1])

                    prob = sess.run(predicted_motion_features, feed_dict={acoustic: acoustic_in,
                                                                                              temporal: temporal_in,
                                                                                              mask: mask_in})
                    predict = prob.reshape((-1,  self.motion_dim))
                    test_predict = np.append(test_predict, predict, axis=0)
                result_test_predict.append(test_predict)

            min_length=999999
            for i in range(len(result_test_predict)):
                min_length=min(min_length,result_test_predict[i].shape[0])

            final_res=result_test_predict[0][self.time_step:min_length,:]
            for i in range(1,len(result_test_predict)):
                final_res+=result_test_predict[i][self.time_step:min_length,:]
            final_res/=len(result_test_predict)
            final_res=np.append(result_test_predict[0][:self.time_step,:],final_res,0)

            test_predict = train_motion_scaler.inverse_transform(final_res)

            test_predict = np.reshape(test_predict, [-1, self.motion_dim//3, 3])
            length = test_predict.shape[0]
            test_predict=self.smooth_skeleton(test_predict)
            test_predict = test_predict.tolist()
            center=center
            data = {"length": length, "skeletons": test_predict,"center":center}
            with open(os.path.join(result_save_dir,file_name), 'w') as file_object:
                json.dump(data, file_object)
            return test_predict
