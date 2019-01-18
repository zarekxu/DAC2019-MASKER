
import numpy as np
import tensorflow as tf
import argparse

import scipy.io.wavfile as wav
# from tensorflow.python.client import timeline

import time
import os
import sys
sys.path.append("DeepSpeech")
np.set_printoptions(threshold=np.nan)

tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
import DeepSpeech
os.path.exists = tmp

from text import ctc_label_dense_to_sparse
from tf_logits import get_logits, compute_mfcc

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

class Attack:
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate, num_iterations=1000 , batch_size=1):
        """
        Set up the attack procedure.

        Here we create the TF graph that we're going to use to
        actually generate the adversarial examples.
        """
        
        self.sess = sess
        self.learning_rate = int(learning_rate)
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.phrase_length = phrase_length
        self.max_audio_len = max_audio_len

        # Create all the variables necessary
        # they are prefixed with qq_ just so that we know hich
        # ones are ours so when we restore the session we don't
        # clobber them.
        self.delta = delta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
        self.mask = mask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
        self.cwmask = cwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
        self.original = original = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
        # self.target = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original_target') #---------------------------target variable--------------------------#
        self.lengths = lengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
        self.importance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')
        self.target_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
        self.target_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
        self.rescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')

        # Initially we bound the l_infty norm by 2000, increase this
        # constant if it's not big enough of a distortion for your dataset.
        self.apply_delta = tf.clip_by_value(delta, -500, 500)

        # We set the new input to the model to be the abve delta
        # plus a mask, which allows us to enforce that certain
        # values remain constant 0 for length padding sequences.
        self.new_input = new_input = self.apply_delta*mask + original         #mask ?????

        # We add a tiny bit of noise to help make sure that we can
        # clip our values to 16-bit integers and not break things.
        noise = tf.random_normal(new_input.shape,stddev=2)
        # pass_in = tf.clip_by_value(new_input+noise, -2**15, 2**15-1)
        pass_in = tf.clip_by_value(new_input, -2**15, 2**15-1)
        # pass_in = tf.clip_by_value(new_input, -2 ** 15, 2 ** 15 - 1)

        # Feed this final value to get the logits.
        self.logits = logits = get_logits(pass_in, lengths)


        #----------------------------Feature Loss-------------------------------------#
        self.feature = feature = compute_mfcc(pass_in)
        self.target_feature = tf.zeros(self.feature.shape)

        #---------------------------------------------------------------------------

        #print(self.feature)
        # dic=tf.argmax(self.feature,2)
        # print(dic)
        #dic=tf.nn.top_k(tf.abs(self.feature),10)
        #dic=tf.negative(tf.nn.top_k(tf.negative(self.feature),1))

        #print(dic[0])
        #self.feature=dic[0]
        #print(self.feature)
        #self.feature=tf.stop_gradient(self.feature)

        #------------------------------------------------------------------------

        # And finally restore the graph to make the classifier
        # actually do something interesting.
        saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
        saver.restore(sess, "models/session_dump")

        # Choose the loss function we want -- either CTC or CW
        self.loss_fn = loss_fn
        if loss_fn == "CTC":
            target = ctc_label_dense_to_sparse(self.target_phrase, self.target_phrase_lengths, batch_size)
            
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32),
                                     inputs=logits, sequence_length=lengths)
            
            loss = tf.nn.relu(ctcloss)
            
        elif loss_fn == "CW":
            raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
        elif loss_fn == "MFCC":
            mfcc_loss = tf.losses.mean_squared_error(self.feature, self.target_feature)
            loss = tf.nn.relu(mfcc_loss)
        elif loss_fn == "NOISE":
            self.target_feature = self.feature
            loss = tf.losses.mean_squared_error(self.feature, self.target_feature)
        else:
            raise
    
        # Set up the Adam optimizer to perform gradient descent for us
        var_start = tf.global_variables()
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=[delta])
        self.loss = loss
        #self.ctcloss = ctcloss

        
        var_end = tf.global_variables()
        new_vars = [x for x in var_end if x.name not in [y.name for y in var_start]]
        sess.run(tf.variables_initializer(new_vars+[delta]))

        # Decoder from the logits, to see how we're doing
        self.decoded, _ = tf.nn.ctc_beam_search_decoder(logits, lengths, merge_repeated=False, beam_width=1000)

    def attack(self, loss, audio, lengths, target,  i=None, j=None):
        sess = self.sess

        # Initialize all of the variables
        # TODO: each of these assign ops creates a new TF graph
        # object, and they should be all created only once in the
        # constructor. It works fine as long as you don't call
        # attack() a bunch of times.
        sess.run(tf.variables_initializer([self.delta]))
        sess.run(self.original.assign(np.array(audio)))
        # sess.run(self.target.assign(np.array(target_audio)))                              #-------------------------------target feed value---------------------------#
        sess.run(self.lengths.assign((np.array(lengths)-1)//320))  #320 sampling point for each frame
        sess.run(self.mask.assign(np.array([[1 if i < l else 0 for i in range(self.max_audio_len)] for l in lengths])))
        sess.run(self.cwmask.assign(np.array([[1 if i < l else 0 for i in range(self.phrase_length)] for l in (np.array(lengths)-1)//320])))
        sess.run(self.target_phrase_lengths.assign(np.array([len(x) for x in target])))
        sess.run(self.target_phrase.assign(np.array([list(t)+[0]*(self.phrase_length-len(t)) for t in target])))
        c = np.ones((self.batch_size, self.phrase_length))
        sess.run(self.importance.assign(c))
        sess.run(self.rescale.assign(np.ones((self.batch_size,1))))

        # Here we'll keep track of the best solution we've found so far
        final_deltas = [None]*self.batch_size
        real_deltas = [None]*self.batch_size
        #print(real_deltas,'@@@@@@@@@@@')
        # We'll make a bunch of iterations of gradient descent here
        MAX = self.num_iterations     #iteration number
        for i in range(MAX):
            # Print out some debug information every 10 iterations.
            if i%1 == 0:
                new, delta, out, logits = sess.run((self.new_input, self.delta, self.decoded, self.logits))
                #print(out) #dic, each number indicate the symbol in alphatxt.
                #FEATURE=sess.run(self.feature)
                #print(FEATURE)
                #res=np.argsort(FEATURE)
                #print(res)
                #self.feature=tf.stop_gradient(self.feature[:,:,:])
                # self.feature=tf.assign(self.feature,FEATURE)
                # sess.run(self.feature)
                chars = out[0].values
                print('out[0] is :',out[0])
                res = np.zeros(out[0].dense_shape)+len(toks)-1       #dense_shape ?????
                #print(res)
                for ii in range(len(out[0].values)):
                    x,y = out[0].indices[ii]
                    # print('!!!!!')
                    # print('x is :',x)
                    # print('y is :',y)
                    res[x,y] = out[0].values[ii]

                # Here we print the strings that are recognized.
                res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res]
                print("\n".join(res),'!!!!!!!!!!!!!!!')

                # And here we print the argmax of the alignment.
                res2 = np.argmax(logits,axis=2).T
                res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,lengths)]
                print("\n".join(res2),'!!!!!!!!!!!!!!!')

            # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()
            #pre = time.time()
            # Actually do the optimization ste
            # d, el, l, logits, new_input = sess.run((self.delta, self.expanded_loss, self.loss, self.logits, self.new_input))

            #pre = time.time()

            # d, new_input, l, _ = sess.run((self.delta, self.new_input, self.loss, self.train), options = options, run_metadata = run_metadata)
            if loss == "NOISE":
                sess.run(self.delta.assign(tf.random_normal(self.delta.shape, mean=0, stddev=300)))
                pre = time.time()
                new_input, l = sess.run((self.new_input, self.loss))

            else:
            #   d, new_input, l, _ = sess.run((self.delta, self.new_input, self.loss, self.train))
                d= sess.run(self.delta)
                new_input= sess.run(self.new_input)
                l = sess.run(self.loss)
                pre = time.time()
                sess.run(self.train)


            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_0'+str(i)+'.json', 'w') as f:
            #     f.write(chrome_trace)
            print(i," Iteration time:", time.time() - pre)

            ii=0
            # rescale = sess.run(self.rescale)
            # rescale[ii] *= .8
            # print(np.array(audio).shape)
            # print(np.array(audio[ii]).shape)
            real_deltas[ii] = new_input[ii] - audio[ii]
            # print(real_deltas[0])
            print("noise DB:", 20 * np.log10(np.max(np.abs(real_deltas[ii])+1)))
            # Report progress
            # print(np.mean(cl), np.mean(l), "\t".join("%.3f"%x for x in cl))

            logits = np.argmax(logits,axis=2).T
            #print(self.batch_size)
            for ii in range(self.batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                real_deltas[ii] = new_input[ii] - audio[ii]
                if (i%100 == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                   or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    # rescale = sess.run(self.rescale)
                    # if rescale[ii]*2000 > np.max(np.abs(d)):
                    #     # If we're already below the threshold, then
                    #     # just reduce the threshold to the current
                    #     # point and save some time.
                    #     # print("It's way over", np.max(np.abs(d))/2000.0)
                    #     # print("noise DB:", 20*np.log10(np.max(np.abs(real_deltas[ii]))))
                    #     rescale[ii] = np.max(np.abs(d))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.


                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]


                    # print("Worked",ii,cl[ii],rescale[ii])
                    # print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    # sess.run(self.rescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    # wav.write("/tmp/adv.wav", 16000,
                    #           np.array(np.clip(np.round(new_input[ii]),
                    #                            -2**15, 2**15-1),dtype=np.int16))
        #print("\n".join(res))
        return real_deltas, final_deltas


    
def main():
    """
    Do the attack here.

    This is all just boilerplate; nothing interesting
    happens in this method.

    For now we only support using CTC loss and only generating
    one adversarial example at a time.
    """
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('--in', type=str, dest="input",
    #                     required=True,
    #                     help="Input audio .wav file, at 16KHz")
    # parser.add_argument('--target', type=str,
    #                     required=True,
    #                     help="Target transcription")
    # parser.add_argument('--out', type=str, dest="output",
    #                     required=True,
    #                     help="Where to put the adversarial example")
    # parser.add_argument('--loss', type=str, dest="loss",
    #                     required=True,
    #                     help="What loss to use to craft adversarial example")
    # parser.add_argument('--lr', type=str,
    #                     required=True,
    #                     help="How much is lr")
    # args = parser.parse_args()

    deltas_array = []


    input = "sample_1.wav"
    out = "s_ctc.wav"
    loss = "CTC"
    target_audio = "./commonvoice_subset/sample-000022.wav"
    lr = 80
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        audios = []
        targets = []
        lengths = []

        # Just load one input
        for i in range(1):
            fs, audio = wav.read(input)
            #fs, audio = wav.read(args.input)
            # _, target = wav.read(target_audio)
            assert fs == 16000
            print('source dB', 20*np.log10(np.max(np.abs(audio))))
            audios.append(list(audio))
            # targets.append(list(target))
            lengths.append(len(audio))
        audios = np.array(audios)
        # if len(audio) < len(target):
        #     target = np.array(targets[:,0:len(audios[0])])
        # else:
        #     targets[0].extend([0 for i in range(len(audio) - len(target))])
        #     target = np.array(targets)
        maxlen = len(audios[0])

        phrase = "open the door"

        # Set up the attack class and run it
        attack = Attack(sess, loss, len(phrase), maxlen, learning_rate=lr, batch_size=len(audios))
        real_deltas, deltas = attack.attack(loss, audios, lengths,
                               [[toks.index(x) for x in phrase]]*len(audios))

        # deltas_array.append(real_deltas)

        # And now save it to the desired output
        for i in range(1):
            wav.write(out, 16000,
                      np.array(np.clip(np.round(deltas[i][:lengths[i]]),
                                           -2**15, 2**15-1),dtype=np.int16))
            # wav.write(args.perturbation, 16000,
            #           np.array(np.clip(np.round(real_deltas[i][:lengths[i]]),
            #                                -2**15, 2**15-1),dtype=np.int16))

main()
