import tensorflow as tf
from data_load import load_vocab
from modules import get_token_embeddings, multihead_attention, ff, positional_encoding

class Transformer(object):
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)

        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)
        print(self.embeddings)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)

            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries = enc,
                                              keys = enc,
                                              values = enc,
                                              num_heads = self.hp.num_heads,
                                              dropout_rate = self.hp.dropout_rate,
                                              training = training,
                                              causality = False)

                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])

            memory = enc

            return memory, sents1

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1 = self.encode(xs)

        return None, None, None, None
