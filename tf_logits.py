from python_speech_features import mfcc
import math
import os
import sys
sys.path.append("DeepSpeech")

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav

# Okay, so this is ugly. We don't want DeepSpeech to crash
# when we haven't built the language model.
# So we're just going to monkeypatch TF and make it a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
import DeepSpeech
os.path.exists = tmp

#-------refer from Audio Adversarial Example--------
def compute_mfcc(audio, **kwargs):
    """
    Compute the MFCC for a given audio waveform. This is
    identical to how DeepSpeech does it, but does it all in
    TensorFlow so that we can differentiate through it.
    """

    batch_size, size = audio.get_shape().as_list()
    #print(audio)
    audio = tf.cast(audio, tf.float32)
    #print('!!!!!')
    #print(audio)       #audio is [1,N]
    # 1. Pre-emphasizer, a high-pass filter
    # audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1], np.zeros((batch_size,1000),dtype=np.float32)), 1) #[:,1:] means from 1st to last one   [:,:-1] means all except last one
    audio = tf.concat((audio[:, :1], audio[:, 1:] - 0.97*audio[:, :-1]), 1)
    # 2. windowing into frames of 320 samples, overlapping
    windowed = tf.stack([audio[:, i:i+320] for i in range(0,size-320,160)],1)   #overlapping 160
    #print(windowed.get_shape())
    # for i in range(windowed.shape[1]):
    #     for j in range(windowed.shape[2]):
    #         #print(j)
    #         output_list=[]
    #         output_list.append(tf.multiply(windowed[:,:,j],(0.54-0.46*math.cos(2*3.14*(j-1)/319))))
    # windowed=tf.stack(output_list)
    # 3. Take the FFT to convert to frequency space
    ffted = tf.spectral.rfft(windowed, [512])
    ffted = 1.0 / 512 * tf.square(tf.abs(ffted))
    #print('fft::',ffted)
   #print(ffted.get_shape())
#-------------------------------
    feat=ffted
    print("cccccccccc::",feat.get_shape())
    FEATURE=tf.stop_gradient(feat[:,:,0:160])
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaa",FEATURE.get_shape())
    FEATURE1=feat[:,:,160:]
    print("bbbbbb::",FEATURE1.get_shape())
    FEATURE=tf.concat((FEATURE,FEATURE1),axis=2)
    print(FEATURE)
    feature1=FEATURE
    ffted=feature1





#--------------------------------
    # feat=ffted
    # a=feat.get_shape()
    #
    # dic=tf.nn.top_k(feat,20)
    # print(dic)
    # #dic=tf.cast(dic, tf.int32)
    # dic1=dic[1]
    # print('dic1:',dic1)
    # b1=dic1[0,:,:]
    # print('b1:',b1)
    # # b2=b1[0]
    # # print(b2)
    # # print(b2[0])
    # print('------')
    #
    #
    # b2=b1[0]
    # b2=tf.nn.top_k(b2,20)
    # b2=b2[0]
    # print(b2)
    # FEATURE1=tf.stop_gradient(feat[:,0,b2[19]])
    # #print(FEATURE1)
    # FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    # # print(FEATURE1)
    # FEATURE2=tf.stop_gradient(feat[:,0,b2[18]])
    # FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    # FEATURE3=tf.stop_gradient(feat[:,0,b2[17]])
    # FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    # FEATURE4=tf.stop_gradient(feat[:,0,b2[16]])
    # FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    # FEATURE5=tf.stop_gradient(feat[:,0,b2[15]])
    # FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    # FEATURE6=tf.stop_gradient(feat[:,0,b2[14]])
    # FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    # FEATURE7=tf.stop_gradient(feat[:,0,b2[13]])
    # FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    # FEATURE8=tf.stop_gradient(feat[:,0,b2[12]])
    # FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    # FEATURE9=tf.stop_gradient(feat[:,0,b2[11]])
    # FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    # FEATURE10=tf.stop_gradient(feat[:,0,b2[10]])
    # FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    # FEATURE11=tf.stop_gradient(feat[:,0,b2[9]])
    # FEATURE11=tf.reshape(FEATURE11,[1,-1,1])
    # FEATURE12=tf.stop_gradient(feat[:,0,b2[8]])
    # FEATURE12=tf.reshape(FEATURE12,[1,-1,1])
    # FEATURE13=tf.stop_gradient(feat[:,0,b2[7]])
    # FEATURE13=tf.reshape(FEATURE13,[1,-1,1])
    # FEATURE14=tf.stop_gradient(feat[:,0,b2[6]])
    # FEATURE14=tf.reshape(FEATURE14,[1,-1,1])
    # FEATURE15=tf.stop_gradient(feat[:,0,b2[5]])
    # FEATURE15=tf.reshape(FEATURE15,[1,-1,1])
    # FEATURE16=tf.stop_gradient(feat[:,0,b2[4]])
    # FEATURE16=tf.reshape(FEATURE16,[1,-1,1])
    # FEATURE17=tf.stop_gradient(feat[:,0,b2[3]])
    # FEATURE17=tf.reshape(FEATURE17,[1,-1,1])
    # FEATURE18=tf.stop_gradient(feat[:,0,b2[2]])
    # FEATURE18=tf.reshape(FEATURE18,[1,-1,1])
    # FEATURE19=tf.stop_gradient(feat[:,0,b2[1]])
    # FEATURE19=tf.reshape(FEATURE19,[1,-1,1])
    # FEATURE20=tf.stop_gradient(feat[:,0,b2[0]])
    # FEATURE20=tf.reshape(FEATURE20,[1,-1,1])
    #
    #
    # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    # FEATURE0001=feat[:,0,0:b2[19]]
    # FEATURE0001=tf.reshape(FEATURE0001,[1,1,-1])
    # FEATURE0102=feat[:,0,b2[19]+1:b2[18]]
    # FEATURE0102=tf.reshape(FEATURE0102,[1,1,-1])
    # FEATURE0203=feat[:,0,b2[18]+1:b2[17]]
    # FEATURE0203=tf.reshape(FEATURE0203,[1,1,-1])
    # FEATURE0304=feat[:,0,b2[17]+1:b2[16]]
    # FEATURE0304=tf.reshape(FEATURE0304,[1,1,-1])
    # FEATURE0405=feat[:,0,b2[16]+1:b2[15]]
    # FEATURE0405=tf.reshape(FEATURE0405,[1,1,-1])
    # FEATURE0506=feat[:,0,b2[15]+1:b2[14]]
    # FEATURE0506=tf.reshape(FEATURE0506,[1,1,-1])
    # FEATURE0607=feat[:,0,b2[14]+1:b2[13]]
    # FEATURE0607=tf.reshape(FEATURE0607,[1,1,-1])
    # FEATURE0708=feat[:,0,b2[13]+1:b2[12]]
    # FEATURE0708=tf.reshape(FEATURE0708,[1,1,-1])
    # FEATURE0809=feat[:,0,b2[12]+1:b2[11]]
    # FEATURE0809=tf.reshape(FEATURE0809,[1,1,-1])
    # FEATURE0910=feat[:,0,b2[11]+1:b2[10]]
    # FEATURE0910=tf.reshape(FEATURE0910,[1,1,-1])
    # FEATURE1011=feat[:,0,b2[10]+1:b2[9]]
    # FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    # FEATURE1112=feat[:,0,b2[9]+1:b2[8]]
    # FEATURE1112=tf.reshape(FEATURE1112,[1,1,-1])
    # FEATURE1213=feat[:,0,b2[8]+1:b2[7]]
    # FEATURE1213=tf.reshape(FEATURE1213,[1,1,-1])
    # FEATURE1314=feat[:,0,b2[7]+1:b2[6]]
    # FEATURE1314=tf.reshape(FEATURE1314,[1,1,-1])
    # FEATURE1415=feat[:,0,b2[6]+1:b2[5]]
    # FEATURE1415=tf.reshape(FEATURE1415,[1,1,-1])
    # FEATURE1516=feat[:,0,b2[5]+1:b2[4]]
    # FEATURE1516=tf.reshape(FEATURE1516,[1,1,-1])
    # FEATURE1617=feat[:,0,b2[4]+1:b2[3]]
    # FEATURE1617=tf.reshape(FEATURE1617,[1,1,-1])
    # FEATURE1718=feat[:,0,b2[3]+1:b2[2]]
    # FEATURE1718=tf.reshape(FEATURE1718,[1,1,-1])
    # FEATURE1819=feat[:,0,b2[2]+1:b2[1]]
    # FEATURE1819=tf.reshape(FEATURE1819,[1,1,-1])
    # FEATURE1920=feat[:,0,b2[1]+1:b2[0]]
    # FEATURE1920=tf.reshape(FEATURE1920,[1,1,-1])
    # FEATURE2021=feat[:,0,b2[0]+1:]
    # FEATURE2021=tf.reshape(FEATURE2021,[1,1,-1])
    #
    #
    # feature=tf.concat((FEATURE0001,FEATURE1,FEATURE0102,FEATURE2,FEATURE0203,FEATURE3,FEATURE0304,FEATURE4,FEATURE0405,FEATURE5,FEATURE0506,FEATURE6,FEATURE0607,FEATURE7,FEATURE0708,FEATURE8,FEATURE0809,FEATURE9,FEATURE0910,FEATURE10,FEATURE1011,FEATURE11,FEATURE1112,FEATURE12,FEATURE1213,FEATURE13,FEATURE1314,FEATURE14,FEATURE1415,FEATURE15,FEATURE1516,FEATURE16,FEATURE1617,FEATURE17,FEATURE1718,FEATURE18,FEATURE1819,FEATURE19,FEATURE1920,FEATURE20,FEATURE2021),axis=2)
    # #feature=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    # print(feature)
    # print('######')
    #
    # for i in range(1,a[1]):
    #     # FEATURE=tf.stop_gradient(feat[:,i,b1[i]])
    #     # FEATURE=tf.reshape(FEATURE,[1,-1,1])
    #     # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     # FEATURE1=feat[:,i,0:b1[i]]
    #     # FEATURE1=tf.reshape(FEATURE1,[1,1,-1])
    #     # #print(FEATURE1)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE2=feat[:,i,b1[i]+1:26]
    #     # FEATURE2=tf.reshape(FEATURE2,[1,1,-1])
    #     # #print(FEATURE2)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE=tf.concat((FEATURE1,FEATURE,FEATURE2),axis=2)
    #     #print(FEATURE)
    #     b2=b1[i]
    #     b2=tf.nn.top_k(b2,20)
    #     b2=b2[0]
    #
    #     FEATURE1=tf.stop_gradient(feat[:,i,b2[19]])
    #     #print(FEATURE1)
    #     FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    #     # print(FEATURE1)
    #     FEATURE2=tf.stop_gradient(feat[:,i,b2[18]])
    #     FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    #     FEATURE3=tf.stop_gradient(feat[:,i,b2[17]])
    #     FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    #     FEATURE4=tf.stop_gradient(feat[:,i,b2[16]])
    #     FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    #     FEATURE5=tf.stop_gradient(feat[:,i,b2[15]])
    #     FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    #     FEATURE6=tf.stop_gradient(feat[:,i,b2[14]])
    #     FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    #     FEATURE7=tf.stop_gradient(feat[:,i,b2[13]])
    #     FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    #     FEATURE8=tf.stop_gradient(feat[:,i,b2[12]])
    #     FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    #     FEATURE9=tf.stop_gradient(feat[:,i,b2[11]])
    #     FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    #     FEATURE10=tf.stop_gradient(feat[:,i,b2[10]])
    #     FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    #     FEATURE11=tf.stop_gradient(feat[:,i,b2[9]])
    #     FEATURE11=tf.reshape(FEATURE11,[1,-1,1])
    #     FEATURE12=tf.stop_gradient(feat[:,i,b2[8]])
    #     FEATURE12=tf.reshape(FEATURE12,[1,-1,1])
    #     FEATURE13=tf.stop_gradient(feat[:,i,b2[7]])
    #     FEATURE13=tf.reshape(FEATURE13,[1,-1,1])
    #     FEATURE14=tf.stop_gradient(feat[:,i,b2[6]])
    #     FEATURE14=tf.reshape(FEATURE14,[1,-1,1])
    #     FEATURE15=tf.stop_gradient(feat[:,i,b2[5]])
    #     FEATURE15=tf.reshape(FEATURE15,[1,-1,1])
    #     FEATURE16=tf.stop_gradient(feat[:,i,b2[4]])
    #     FEATURE16=tf.reshape(FEATURE16,[1,-1,1])
    #     FEATURE17=tf.stop_gradient(feat[:,i,b2[3]])
    #     FEATURE17=tf.reshape(FEATURE17,[1,-1,1])
    #     FEATURE18=tf.stop_gradient(feat[:,i,b2[2]])
    #     FEATURE18=tf.reshape(FEATURE18,[1,-1,1])
    #     FEATURE19=tf.stop_gradient(feat[:,i,b2[1]])
    #     FEATURE19=tf.reshape(FEATURE19,[1,-1,1])
    #     FEATURE20=tf.stop_gradient(feat[:,i,b2[0]])
    #     FEATURE20=tf.reshape(FEATURE20,[1,-1,1])
    #
    #
    #
    #     #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     FEATURE0001=feat[:,i,0:b2[19]]
    #     FEATURE0001=tf.reshape(FEATURE0001,[1,1,-1])
    #     FEATURE0102=feat[:,i,b2[19]+1:b2[18]]
    #     FEATURE0102=tf.reshape(FEATURE0102,[1,1,-1])
    #     FEATURE0203=feat[:,i,b2[18]+1:b2[17]]
    #     FEATURE0203=tf.reshape(FEATURE0203,[1,1,-1])
    #     FEATURE0304=feat[:,i,b2[17]+1:b2[16]]
    #     FEATURE0304=tf.reshape(FEATURE0304,[1,1,-1])
    #     FEATURE0405=feat[:,i,b2[16]+1:b2[15]]
    #     FEATURE0405=tf.reshape(FEATURE0405,[1,1,-1])
    #     FEATURE0506=feat[:,i,b2[15]+1:b2[14]]
    #     FEATURE0506=tf.reshape(FEATURE0506,[1,1,-1])
    #     FEATURE0607=feat[:,i,b2[14]+1:b2[13]]
    #     FEATURE0607=tf.reshape(FEATURE0607,[1,1,-1])
    #     FEATURE0708=feat[:,i,b2[13]+1:b2[12]]
    #     FEATURE0708=tf.reshape(FEATURE0708,[1,1,-1])
    #     FEATURE0809=feat[:,i,b2[12]+1:b2[11]]
    #     FEATURE0809=tf.reshape(FEATURE0809,[1,1,-1])
    #     FEATURE0910=feat[:,i,b2[11]+1:b2[10]]
    #     FEATURE0910=tf.reshape(FEATURE0910,[1,1,-1])
    #     FEATURE1011=feat[:,i,b2[10]+1:b2[9]]
    #     FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    #     FEATURE1112=feat[:,i,b2[9]+1:b2[8]]
    #     FEATURE1112=tf.reshape(FEATURE1112,[1,1,-1])
    #     FEATURE1213=feat[:,i,b2[8]+1:b2[7]]
    #     FEATURE1213=tf.reshape(FEATURE1213,[1,1,-1])
    #     FEATURE1314=feat[:,i,b2[7]+1:b2[6]]
    #     FEATURE1314=tf.reshape(FEATURE1314,[1,1,-1])
    #     FEATURE1415=feat[:,i,b2[6]+1:b2[5]]
    #     FEATURE1415=tf.reshape(FEATURE1415,[1,1,-1])
    #     FEATURE1516=feat[:,i,b2[5]+1:b2[4]]
    #     FEATURE1516=tf.reshape(FEATURE1516,[1,1,-1])
    #     FEATURE1617=feat[:,i,b2[4]+1:b2[3]]
    #     FEATURE1617=tf.reshape(FEATURE1617,[1,1,-1])
    #     FEATURE1718=feat[:,i,b2[3]+1:b2[2]]
    #     FEATURE1718=tf.reshape(FEATURE1718,[1,1,-1])
    #     FEATURE1819=feat[:,i,b2[2]+1:b2[1]]
    #     FEATURE1819=tf.reshape(FEATURE1819,[1,1,-1])
    #     FEATURE1920=feat[:,i,b2[1]+1:b2[0]]
    #     FEATURE1920=tf.reshape(FEATURE1920,[1,1,-1])
    #     FEATURE2021=feat[:,i,b2[0]+1:]
    #     FEATURE2021=tf.reshape(FEATURE2021,[1,1,-1])
    #
    #
    #     FEATURE=tf.concat((FEATURE0001,FEATURE1,FEATURE0102,FEATURE2,FEATURE0203,FEATURE3,FEATURE0304,FEATURE4,FEATURE0405,FEATURE5,FEATURE0506,FEATURE6,FEATURE0607,FEATURE7,FEATURE0708,FEATURE8,FEATURE0809,FEATURE9,FEATURE0910,FEATURE10,FEATURE1011,FEATURE11,FEATURE1112,FEATURE12,FEATURE1213,FEATURE13,FEATURE1314,FEATURE14,FEATURE1415,FEATURE15,FEATURE1516,FEATURE16,FEATURE1617,FEATURE17,FEATURE1718,FEATURE18,FEATURE1819,FEATURE19,FEATURE1920,FEATURE20,FEATURE2021),axis=2)
    #     #FEATURE=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    #     feature=tf.concat((feature,FEATURE),axis=1)
    #     #print(feature)
    #
    # # feature=tf.reshape(feature,[1,feat[1],feat[2]])
    # print('!!!!!')
    # print(feature)
    # #feature1=tf.Variable(tf.ones_like(feature),validate_shape=True)
    # feature1=tf.reshape(feature,[1,a[1],257])
    # print(feature1)
    # ffted=feature1

#--------------------------------
    # 4. Compute the Mel windowing of the FFT
    energy = tf.reduce_sum(ffted,axis=2)+1e-30

    filters = np.load("filterbanks.npy").T     #size is (257, 26)
    #print(np.shape(filters))
    feat = tf.matmul(ffted, np.array([filters]*batch_size,dtype=np.float32))+1e-30     #size is (1, 560, 26)
    print('filter bank:::',feat)
    # dic=tf.argmax(feat,2)   #shape=(1, 378)
    #print(dic)
    #dic=tf.nn.top_k(feat,10)
    #print(dic[0])
    #print('!!!')
    #b=dic.get_shape()
    #print(feat.shape[1])
    #print(dic[0,1])

#----------------------------------------------------------------------
    # #
    # FEATURE=tf.stop_gradient(feat[:,:,0:25])
    # #print(FEATURE.get_shape())
    # FEATURE1=feat[:,:,25:]
    # #print(FEATURE1.get_shape())
    # FEATURE=tf.concat((FEATURE,FEATURE1),axis=2)
    # print(FEATURE)
    # feature1=FEATURE
#----------------------------------------------------------------------
    # a=feat.get_shape()
    # #print(a)
    # # dic = tf.Variable(tf.zeros(shape=[1,a[1]]), name='dic')
    # # print(dic)
    # dic=tf.argmax(feat,2)
    # dic=tf.cast(dic, tf.int32)
    # # dic=dic1
    # # print(dic[0])
    # # aa=dic[0]
    # # print(aa[1])
    # # dic=tf.Session().run(dic)
    # #print(dic)  #Tensor("ArgMax:0", shape=(1, 378), dtype=int64)
    # b1=dic[0]
    # #print(b1)  #Tensor("strided_slice_381:0", shape=(378,), dtype=int64)
    # b2=b1[0]
    # print(b2)
    # # b22=b2[2]
    # # print(b1[:,1,:])
    # # feature=[]
    # # feature=tf.zeros([0,0,0],tf.float32)
    # FEATURE=tf.stop_gradient(feat[:,0,b1[0]])
    # FEATURE=tf.reshape(FEATURE,[1,-1,1])
    # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    # FEATURE1=feat[:,0,0:b1[0]]
    # FEATURE1=tf.reshape(FEATURE1,[1,1,-1])
    # #print(FEATURE1)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    # FEATURE2=feat[:,0,b1[0]+1:26]
    # FEATURE2=tf.reshape(FEATURE2,[1,1,-1])
    # #print(FEATURE2)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    # feature=tf.concat((FEATURE1,FEATURE,FEATURE2),axis=2)
    # #print(feature)
    #
    #
    # for i in range(1,a[1]):
    #     FEATURE=tf.stop_gradient(feat[:,i,b1[i]])
    #     FEATURE=tf.reshape(FEATURE,[1,-1,1])
    #     #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     FEATURE1=feat[:,i,0:b1[i]]
    #     FEATURE1=tf.reshape(FEATURE1,[1,1,-1])
    #     #print(FEATURE1)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     FEATURE2=feat[:,i,b1[i]+1:26]
    #     FEATURE2=tf.reshape(FEATURE2,[1,1,-1])
    #     #print(FEATURE2)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     FEATURE=tf.concat((FEATURE1,FEATURE,FEATURE2),axis=2)
    #     #print(FEATURE)
    #     feature=tf.concat((feature,FEATURE),axis=1)
    #     #print(feature)
    #
    # # feature=tf.reshape(feature,[1,feat[1],feat[2]])
    # print('!!!!!')
    # print(feature)
    # #feature1=tf.Variable(tf.ones_like(feature),validate_shape=True)
    # feature1=tf.reshape(feature,[1,a[1],26])
    # print(feature1)

#----------------------------------------------------------------------------------

    # a=feat.get_shape()
    #
    # dic=tf.nn.top_k(feat,10)
    # print(dic)
    # #dic=tf.cast(dic, tf.int32)
    # dic1=dic[1]
    # print('dic1:',dic1)
    # b1=dic1[0,:,:]
    # print('b1:',b1)
    # # b2=b1[0]
    # # print(b2)
    # # print(b2[0])
    # print('------')
    #
    #
    # b2=b1[0]
    # b2=tf.nn.top_k(b2,10)
    # b2=b2[0]
    # print(b2)
    # FEATURE1=tf.stop_gradient(feat[:,0,b2[9]])
    # #print(FEATURE1)
    # FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    # # print(FEATURE1)
    # FEATURE2=tf.stop_gradient(feat[:,0,b2[8]])
    # FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    # FEATURE3=tf.stop_gradient(feat[:,0,b2[7]])
    # FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    # FEATURE4=tf.stop_gradient(feat[:,0,b2[6]])
    # FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    # FEATURE5=tf.stop_gradient(feat[:,0,b2[5]])
    # FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    # FEATURE6=tf.stop_gradient(feat[:,0,b2[4]])
    # FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    # FEATURE7=tf.stop_gradient(feat[:,0,b2[3]])
    # FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    # FEATURE8=tf.stop_gradient(feat[:,0,b2[2]])
    # FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    # FEATURE9=tf.stop_gradient(feat[:,0,b2[1]])
    # FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    # FEATURE10=tf.stop_gradient(feat[:,0,b2[0]])
    # FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    #
    # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    # FEATURE01=feat[:,0,0:b2[9]]
    # FEATURE01=tf.reshape(FEATURE01,[1,1,-1])
    # FEATURE12=feat[:,0,b2[9]+1:b2[8]]
    # FEATURE12=tf.reshape(FEATURE12,[1,1,-1])
    # FEATURE23=feat[:,0,b2[8]+1:b2[7]]
    # FEATURE23=tf.reshape(FEATURE23,[1,1,-1])
    # FEATURE34=feat[:,0,b2[7]+1:b2[6]]
    # FEATURE34=tf.reshape(FEATURE34,[1,1,-1])
    # FEATURE45=feat[:,0,b2[6]+1:b2[5]]
    # FEATURE45=tf.reshape(FEATURE45,[1,1,-1])
    # FEATURE56=feat[:,0,b2[5]+1:b2[4]]
    # FEATURE56=tf.reshape(FEATURE56,[1,1,-1])
    # FEATURE67=feat[:,0,b2[4]+1:b2[3]]
    # FEATURE67=tf.reshape(FEATURE67,[1,1,-1])
    # FEATURE78=feat[:,0,b2[3]+1:b2[2]]
    # FEATURE78=tf.reshape(FEATURE78,[1,1,-1])
    # FEATURE89=feat[:,0,b2[2]+1:b2[1]]
    # FEATURE89=tf.reshape(FEATURE89,[1,1,-1])
    # FEATURE910=feat[:,0,b2[1]+1:b2[0]]
    # FEATURE910=tf.reshape(FEATURE910,[1,1,-1])
    # FEATURE1011=feat[:,0,b2[0]+1:26]
    # FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    #
    # feature=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56,FEATURE6,FEATURE67,FEATURE7,FEATURE78,FEATURE8,FEATURE89,FEATURE9,FEATURE910,FEATURE10,FEATURE1011),axis=2)
    # #feature=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    # print(feature)
    # print('######')
    #
    # for i in range(1,a[1]):
    #     # FEATURE=tf.stop_gradient(feat[:,i,b1[i]])
    #     # FEATURE=tf.reshape(FEATURE,[1,-1,1])
    #     # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     # FEATURE1=feat[:,i,0:b1[i]]
    #     # FEATURE1=tf.reshape(FEATURE1,[1,1,-1])
    #     # #print(FEATURE1)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE2=feat[:,i,b1[i]+1:26]
    #     # FEATURE2=tf.reshape(FEATURE2,[1,1,-1])
    #     # #print(FEATURE2)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE=tf.concat((FEATURE1,FEATURE,FEATURE2),axis=2)
    #     #print(FEATURE)
    #     b3=b1[i]
    #     b3=tf.nn.top_k(b3,10)
    #     b3=b3[0]
    #
    #
    #     FEATURE1=tf.stop_gradient(feat[:,0,b3[9]])
    #     #print(FEATURE1)
    #     FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    #     # print(FEATURE1)
    #     FEATURE2=tf.stop_gradient(feat[:,i,b3[8]])
    #     FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    #     FEATURE3=tf.stop_gradient(feat[:,i,b3[7]])
    #     FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    #     FEATURE4=tf.stop_gradient(feat[:,i,b3[6]])
    #     FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    #     FEATURE5=tf.stop_gradient(feat[:,i,b3[5]])
    #     FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    #     FEATURE6=tf.stop_gradient(feat[:,i,b3[4]])
    #     FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    #     FEATURE7=tf.stop_gradient(feat[:,i,b3[3]])
    #     FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    #     FEATURE8=tf.stop_gradient(feat[:,i,b3[2]])
    #     FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    #     FEATURE9=tf.stop_gradient(feat[:,i,b3[1]])
    #     FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    #     FEATURE10=tf.stop_gradient(feat[:,i,b3[0]])
    #     FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    #     #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     FEATURE01=feat[:,i,0:b3[9]]
    #     FEATURE01=tf.reshape(FEATURE01,[1,1,-1])
    #     FEATURE12=feat[:,i,b3[9]+1:b3[8]]
    #     FEATURE12=tf.reshape(FEATURE12,[1,1,-1])
    #     FEATURE23=feat[:,i,b3[8]+1:b3[7]]
    #     FEATURE23=tf.reshape(FEATURE23,[1,1,-1])
    #     FEATURE34=feat[:,i,b3[7]+1:b3[6]]
    #     FEATURE34=tf.reshape(FEATURE34,[1,1,-1])
    #     FEATURE45=feat[:,i,b3[6]+1:b3[5]]
    #     FEATURE45=tf.reshape(FEATURE45,[1,1,-1])
    #     FEATURE56=feat[:,i,b3[5]+1:b3[4]]
    #     FEATURE56=tf.reshape(FEATURE56,[1,1,-1])
    #     FEATURE67=feat[:,i,b3[4]+1:b3[3]]
    #     FEATURE67=tf.reshape(FEATURE67,[1,1,-1])
    #     FEATURE78=feat[:,i,b3[3]+1:b3[2]]
    #     FEATURE78=tf.reshape(FEATURE78,[1,1,-1])
    #     FEATURE89=feat[:,i,b3[2]+1:b3[1]]
    #     FEATURE89=tf.reshape(FEATURE89,[1,1,-1])
    #     FEATURE910=feat[:,i,b3[1]+1:b3[0]]
    #     FEATURE910=tf.reshape(FEATURE910,[1,1,-1])
    #     FEATURE1011=feat[:,i,b3[0]+1:26]
    #     FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    #     FEATURE=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56,FEATURE6,FEATURE67,FEATURE7,FEATURE78,FEATURE8,FEATURE89,FEATURE9,FEATURE910,FEATURE10,FEATURE1011),axis=2)
    #     #FEATURE=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    #     feature=tf.concat((feature,FEATURE),axis=1)
    #     #print(feature)
    #
    # # feature=tf.reshape(feature,[1,feat[1],feat[2]])
    # print('!!!!!')
    # print(feature)
    # #feature1=tf.Variable(tf.ones_like(feature),validate_shape=True)
    # feature1=tf.reshape(feature,[1,a[1],26])
    # print(feature1)
#-----------------------------------------------------------------------
    # a=feat.get_shape()
    #
    # dic=tf.nn.top_k(feat,20)
    # print(dic)
    # #dic=tf.cast(dic, tf.int32)
    # dic1=dic[1]
    # print('dic1:',dic1)
    # b1=dic1[0,:,:]
    # print('b1:',b1)
    # # b2=b1[0]
    # # print(b2)
    # # print(b2[0])
    # print('------')
    #
    #
    # b2=b1[0]
    # b2=tf.nn.top_k(b2,20)
    # b2=b2[0]
    # print(b2)
    # FEATURE1=tf.stop_gradient(feat[:,0,b2[19]])
    # #print(FEATURE1)
    # FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    # # print(FEATURE1)
    # FEATURE2=tf.stop_gradient(feat[:,0,b2[18]])
    # FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    # FEATURE3=tf.stop_gradient(feat[:,0,b2[17]])
    # FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    # FEATURE4=tf.stop_gradient(feat[:,0,b2[16]])
    # FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    # FEATURE5=tf.stop_gradient(feat[:,0,b2[15]])
    # FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    # FEATURE6=tf.stop_gradient(feat[:,0,b2[14]])
    # FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    # FEATURE7=tf.stop_gradient(feat[:,0,b2[13]])
    # FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    # FEATURE8=tf.stop_gradient(feat[:,0,b2[12]])
    # FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    # FEATURE9=tf.stop_gradient(feat[:,0,b2[11]])
    # FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    # FEATURE10=tf.stop_gradient(feat[:,0,b2[10]])
    # FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    # FEATURE11=tf.stop_gradient(feat[:,0,b2[9]])
    # FEATURE11=tf.reshape(FEATURE11,[1,-1,1])
    # FEATURE12=tf.stop_gradient(feat[:,0,b2[8]])
    # FEATURE12=tf.reshape(FEATURE12,[1,-1,1])
    # FEATURE13=tf.stop_gradient(feat[:,0,b2[7]])
    # FEATURE13=tf.reshape(FEATURE13,[1,-1,1])
    # FEATURE14=tf.stop_gradient(feat[:,0,b2[6]])
    # FEATURE14=tf.reshape(FEATURE14,[1,-1,1])
    # FEATURE15=tf.stop_gradient(feat[:,0,b2[5]])
    # FEATURE15=tf.reshape(FEATURE15,[1,-1,1])
    # FEATURE16=tf.stop_gradient(feat[:,0,b2[4]])
    # FEATURE16=tf.reshape(FEATURE16,[1,-1,1])
    # FEATURE17=tf.stop_gradient(feat[:,0,b2[3]])
    # FEATURE17=tf.reshape(FEATURE17,[1,-1,1])
    # FEATURE18=tf.stop_gradient(feat[:,0,b2[2]])
    # FEATURE18=tf.reshape(FEATURE18,[1,-1,1])
    # FEATURE19=tf.stop_gradient(feat[:,0,b2[1]])
    # FEATURE19=tf.reshape(FEATURE19,[1,-1,1])
    # FEATURE20=tf.stop_gradient(feat[:,0,b2[0]])
    # FEATURE20=tf.reshape(FEATURE20,[1,-1,1])
    #
    #
    # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    # FEATURE0001=feat[:,0,0:b2[19]]
    # FEATURE0001=tf.reshape(FEATURE0001,[1,1,-1])
    # FEATURE0102=feat[:,0,b2[19]+1:b2[18]]
    # FEATURE0102=tf.reshape(FEATURE0102,[1,1,-1])
    # FEATURE0203=feat[:,0,b2[18]+1:b2[17]]
    # FEATURE0203=tf.reshape(FEATURE0203,[1,1,-1])
    # FEATURE0304=feat[:,0,b2[17]+1:b2[16]]
    # FEATURE0304=tf.reshape(FEATURE0304,[1,1,-1])
    # FEATURE0405=feat[:,0,b2[16]+1:b2[15]]
    # FEATURE0405=tf.reshape(FEATURE0405,[1,1,-1])
    # FEATURE0506=feat[:,0,b2[15]+1:b2[14]]
    # FEATURE0506=tf.reshape(FEATURE0506,[1,1,-1])
    # FEATURE0607=feat[:,0,b2[14]+1:b2[13]]
    # FEATURE0607=tf.reshape(FEATURE0607,[1,1,-1])
    # FEATURE0708=feat[:,0,b2[13]+1:b2[12]]
    # FEATURE0708=tf.reshape(FEATURE0708,[1,1,-1])
    # FEATURE0809=feat[:,0,b2[12]+1:b2[11]]
    # FEATURE0809=tf.reshape(FEATURE0809,[1,1,-1])
    # FEATURE0910=feat[:,0,b2[11]+1:b2[10]]
    # FEATURE0910=tf.reshape(FEATURE0910,[1,1,-1])
    # FEATURE1011=feat[:,0,b2[10]+1:b2[9]]
    # FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    # FEATURE1112=feat[:,0,b2[9]+1:b2[8]]
    # FEATURE1112=tf.reshape(FEATURE1112,[1,1,-1])
    # FEATURE1213=feat[:,0,b2[8]+1:b2[7]]
    # FEATURE1213=tf.reshape(FEATURE1213,[1,1,-1])
    # FEATURE1314=feat[:,0,b2[7]+1:b2[6]]
    # FEATURE1314=tf.reshape(FEATURE1314,[1,1,-1])
    # FEATURE1415=feat[:,0,b2[6]+1:b2[5]]
    # FEATURE1415=tf.reshape(FEATURE1415,[1,1,-1])
    # FEATURE1516=feat[:,0,b2[5]+1:b2[4]]
    # FEATURE1516=tf.reshape(FEATURE1516,[1,1,-1])
    # FEATURE1617=feat[:,0,b2[4]+1:b2[3]]
    # FEATURE1617=tf.reshape(FEATURE1617,[1,1,-1])
    # FEATURE1718=feat[:,0,b2[3]+1:b2[2]]
    # FEATURE1718=tf.reshape(FEATURE1718,[1,1,-1])
    # FEATURE1819=feat[:,0,b2[2]+1:b2[1]]
    # FEATURE1819=tf.reshape(FEATURE1819,[1,1,-1])
    # FEATURE1920=feat[:,0,b2[1]+1:b2[0]]
    # FEATURE1920=tf.reshape(FEATURE1920,[1,1,-1])
    # FEATURE2021=feat[:,0,b2[0]+1:26]
    # FEATURE2021=tf.reshape(FEATURE2021,[1,1,-1])
    #
    #
    # feature=tf.concat((FEATURE0001,FEATURE1,FEATURE0102,FEATURE2,FEATURE0203,FEATURE3,FEATURE0304,FEATURE4,FEATURE0405,FEATURE5,FEATURE0506,FEATURE6,FEATURE0607,FEATURE7,FEATURE0708,FEATURE8,FEATURE0809,FEATURE9,FEATURE0910,FEATURE10,FEATURE1011,FEATURE11,FEATURE1112,FEATURE12,FEATURE1213,FEATURE13,FEATURE1314,FEATURE14,FEATURE1415,FEATURE15,FEATURE1516,FEATURE16,FEATURE1617,FEATURE17,FEATURE1718,FEATURE18,FEATURE1819,FEATURE19,FEATURE1920,FEATURE20,FEATURE2021),axis=2)
    # #feature=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    # print(feature)
    # print('######')
    #
    # for i in range(1,a[1]):
    #     # FEATURE=tf.stop_gradient(feat[:,i,b1[i]])
    #     # FEATURE=tf.reshape(FEATURE,[1,-1,1])
    #     # #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     # FEATURE1=feat[:,i,0:b1[i]]
    #     # FEATURE1=tf.reshape(FEATURE1,[1,1,-1])
    #     # #print(FEATURE1)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE2=feat[:,i,b1[i]+1:26]
    #     # FEATURE2=tf.reshape(FEATURE2,[1,1,-1])
    #     # #print(FEATURE2)   #Tensor("Reshape_1:0", shape=(1, 1, ?), dtype=float32)
    #     # FEATURE=tf.concat((FEATURE1,FEATURE,FEATURE2),axis=2)
    #     #print(FEATURE)
    #     b2=b1[i]
    #     b2=tf.nn.top_k(b2,20)
    #     b2=b2[0]
    #
    #     FEATURE1=tf.stop_gradient(feat[:,i,b2[19]])
    #     #print(FEATURE1)
    #     FEATURE1=tf.reshape(FEATURE1,[1,-1,1])
    #     # print(FEATURE1)
    #     FEATURE2=tf.stop_gradient(feat[:,i,b2[18]])
    #     FEATURE2=tf.reshape(FEATURE2,[1,-1,1])
    #     FEATURE3=tf.stop_gradient(feat[:,i,b2[17]])
    #     FEATURE3=tf.reshape(FEATURE3,[1,-1,1])
    #     FEATURE4=tf.stop_gradient(feat[:,i,b2[16]])
    #     FEATURE4=tf.reshape(FEATURE4,[1,-1,1])
    #     FEATURE5=tf.stop_gradient(feat[:,i,b2[15]])
    #     FEATURE5=tf.reshape(FEATURE5,[1,-1,1])
    #     FEATURE6=tf.stop_gradient(feat[:,i,b2[14]])
    #     FEATURE6=tf.reshape(FEATURE6,[1,-1,1])
    #     FEATURE7=tf.stop_gradient(feat[:,i,b2[13]])
    #     FEATURE7=tf.reshape(FEATURE7,[1,-1,1])
    #     FEATURE8=tf.stop_gradient(feat[:,i,b2[12]])
    #     FEATURE8=tf.reshape(FEATURE8,[1,-1,1])
    #     FEATURE9=tf.stop_gradient(feat[:,i,b2[11]])
    #     FEATURE9=tf.reshape(FEATURE9,[1,-1,1])
    #     FEATURE10=tf.stop_gradient(feat[:,i,b2[10]])
    #     FEATURE10=tf.reshape(FEATURE10,[1,-1,1])
    #     FEATURE11=tf.stop_gradient(feat[:,i,b2[9]])
    #     FEATURE11=tf.reshape(FEATURE11,[1,-1,1])
    #     FEATURE12=tf.stop_gradient(feat[:,i,b2[8]])
    #     FEATURE12=tf.reshape(FEATURE12,[1,-1,1])
    #     FEATURE13=tf.stop_gradient(feat[:,i,b2[7]])
    #     FEATURE13=tf.reshape(FEATURE13,[1,-1,1])
    #     FEATURE14=tf.stop_gradient(feat[:,i,b2[6]])
    #     FEATURE14=tf.reshape(FEATURE14,[1,-1,1])
    #     FEATURE15=tf.stop_gradient(feat[:,i,b2[5]])
    #     FEATURE15=tf.reshape(FEATURE15,[1,-1,1])
    #     FEATURE16=tf.stop_gradient(feat[:,i,b2[4]])
    #     FEATURE16=tf.reshape(FEATURE16,[1,-1,1])
    #     FEATURE17=tf.stop_gradient(feat[:,i,b2[3]])
    #     FEATURE17=tf.reshape(FEATURE17,[1,-1,1])
    #     FEATURE18=tf.stop_gradient(feat[:,i,b2[2]])
    #     FEATURE18=tf.reshape(FEATURE18,[1,-1,1])
    #     FEATURE19=tf.stop_gradient(feat[:,i,b2[1]])
    #     FEATURE19=tf.reshape(FEATURE19,[1,-1,1])
    #     FEATURE20=tf.stop_gradient(feat[:,i,b2[0]])
    #     FEATURE20=tf.reshape(FEATURE20,[1,-1,1])
    #
    #
    #
    #     #print(FEATURE)   #Tensor("Reshape:0", shape=(1, 1, 1), dtype=float32)
    #     FEATURE0001=feat[:,i,0:b2[19]]
    #     FEATURE0001=tf.reshape(FEATURE0001,[1,1,-1])
    #     FEATURE0102=feat[:,i,b2[19]+1:b2[18]]
    #     FEATURE0102=tf.reshape(FEATURE0102,[1,1,-1])
    #     FEATURE0203=feat[:,i,b2[18]+1:b2[17]]
    #     FEATURE0203=tf.reshape(FEATURE0203,[1,1,-1])
    #     FEATURE0304=feat[:,i,b2[17]+1:b2[16]]
    #     FEATURE0304=tf.reshape(FEATURE0304,[1,1,-1])
    #     FEATURE0405=feat[:,i,b2[16]+1:b2[15]]
    #     FEATURE0405=tf.reshape(FEATURE0405,[1,1,-1])
    #     FEATURE0506=feat[:,i,b2[15]+1:b2[14]]
    #     FEATURE0506=tf.reshape(FEATURE0506,[1,1,-1])
    #     FEATURE0607=feat[:,i,b2[14]+1:b2[13]]
    #     FEATURE0607=tf.reshape(FEATURE0607,[1,1,-1])
    #     FEATURE0708=feat[:,i,b2[13]+1:b2[12]]
    #     FEATURE0708=tf.reshape(FEATURE0708,[1,1,-1])
    #     FEATURE0809=feat[:,i,b2[12]+1:b2[11]]
    #     FEATURE0809=tf.reshape(FEATURE0809,[1,1,-1])
    #     FEATURE0910=feat[:,i,b2[11]+1:b2[10]]
    #     FEATURE0910=tf.reshape(FEATURE0910,[1,1,-1])
    #     FEATURE1011=feat[:,i,b2[10]+1:b2[9]]
    #     FEATURE1011=tf.reshape(FEATURE1011,[1,1,-1])
    #     FEATURE1112=feat[:,i,b2[9]+1:b2[8]]
    #     FEATURE1112=tf.reshape(FEATURE1112,[1,1,-1])
    #     FEATURE1213=feat[:,i,b2[8]+1:b2[7]]
    #     FEATURE1213=tf.reshape(FEATURE1213,[1,1,-1])
    #     FEATURE1314=feat[:,i,b2[7]+1:b2[6]]
    #     FEATURE1314=tf.reshape(FEATURE1314,[1,1,-1])
    #     FEATURE1415=feat[:,i,b2[6]+1:b2[5]]
    #     FEATURE1415=tf.reshape(FEATURE1415,[1,1,-1])
    #     FEATURE1516=feat[:,i,b2[5]+1:b2[4]]
    #     FEATURE1516=tf.reshape(FEATURE1516,[1,1,-1])
    #     FEATURE1617=feat[:,i,b2[4]+1:b2[3]]
    #     FEATURE1617=tf.reshape(FEATURE1617,[1,1,-1])
    #     FEATURE1718=feat[:,i,b2[3]+1:b2[2]]
    #     FEATURE1718=tf.reshape(FEATURE1718,[1,1,-1])
    #     FEATURE1819=feat[:,i,b2[2]+1:b2[1]]
    #     FEATURE1819=tf.reshape(FEATURE1819,[1,1,-1])
    #     FEATURE1920=feat[:,i,b2[1]+1:b2[0]]
    #     FEATURE1920=tf.reshape(FEATURE1920,[1,1,-1])
    #     FEATURE2021=feat[:,i,b2[0]+1:26]
    #     FEATURE2021=tf.reshape(FEATURE2021,[1,1,-1])
    #
    #
    #     FEATURE=tf.concat((FEATURE0001,FEATURE1,FEATURE0102,FEATURE2,FEATURE0203,FEATURE3,FEATURE0304,FEATURE4,FEATURE0405,FEATURE5,FEATURE0506,FEATURE6,FEATURE0607,FEATURE7,FEATURE0708,FEATURE8,FEATURE0809,FEATURE9,FEATURE0910,FEATURE10,FEATURE1011,FEATURE11,FEATURE1112,FEATURE12,FEATURE1213,FEATURE13,FEATURE1314,FEATURE14,FEATURE1415,FEATURE15,FEATURE1516,FEATURE16,FEATURE1617,FEATURE17,FEATURE1718,FEATURE18,FEATURE1819,FEATURE19,FEATURE1920,FEATURE20,FEATURE2021),axis=2)
    #     #FEATURE=tf.concat((FEATURE01,FEATURE1,FEATURE12,FEATURE2,FEATURE23,FEATURE3,FEATURE34,FEATURE4,FEATURE45,FEATURE5,FEATURE56),axis=2)
    #     feature=tf.concat((feature,FEATURE),axis=1)
    #     #print(feature)
    #
    # # feature=tf.reshape(feature,[1,feat[1],feat[2]])
    # print('!!!!!')
    # print(feature)
    # #feature1=tf.Variable(tf.ones_like(feature),validate_shape=True)
    # feature1=tf.reshape(feature,[1,a[1],26])
    # print(feature1)

#----------------------------------------------------------------------

    #for i in range(dic.shape[0]):
    # for j in range(dic.shape[1]):
    #     #print(j)
    #     FEATURE[0,j,dic[0,j]]=tf.stop_gradient(feat[0,j,dic[0,j]])
    #     #print(feat[0,j,dic[0,j]])
    #     #FEATURE=tf.concat(2,FEATURE)
    #     #print(FEATURE.get_shape())
    #feat=FEATURE
    # 5. Take the DCT again, because why not
    feat = tf.log(feat)
    feat = tf.spectral.dct(feat, type=2, norm='ortho')[:,:,:26]
    #print(feat.get_shape())
    # 6. Amplify high frequencies for some reason
    _,nframes,ncoeff = feat.get_shape().as_list()
    n = np.arange(ncoeff)
    lift = 1 + (22/2.)*np.sin(np.pi*n/22)
    feat = lift*feat
    width = feat.get_shape().as_list()[1]
    #print(feat.get_shape())
    #print(feat.get_shape())
    # 7. And now stick the energy next to the features
    #feat1=tf.reshape(tf.log(energy),(-1,width,1))
    #print(feat1.get_shape)

    feat = tf.concat((tf.reshape(tf.log(energy),(-1,width,1)), feat[:, :, 1:]), axis=2)
    #dic=tf.nn.top_k(feat,5)

    print(feat)
    #print(feat.get_shape())
    return feat

                                          
def get_logits(new_input, length, first=[]):
    """
    Compute the logits for a given waveform.

    First, preprocess with the TF version of MFC above,
    and then call DeepSpeech on the features.
    """

    # We need to init DeepSpeech the first time we're called
    if first == []:
        first.append(False)
        # Okay, so this is ugly again.
        # We just want it to not crash.
        tf.app.flags.FLAGS.alphabet_config_path = "DeepSpeech/data/alphabet.txt"
        DeepSpeech.initialize_globals()

    batch_size = new_input.get_shape()[0]

    # 1. Compute the MFCCs for the input audio
    # (this is differentiable with our implementation above)
    empty_context = np.zeros((batch_size, 9, 26), dtype=np.float32)
    new_input_to_mfcc = compute_mfcc(new_input)[:, ::2]
    # print(compute_mfcc(new_input))
    # print(new_input_to_mfcc)
    features = tf.concat((empty_context, new_input_to_mfcc, empty_context), 1)

    # 2. We get to see 9 frames at a time to make our decision,
    # so concatenate them together.
    features = tf.reshape(features, [new_input.get_shape()[0], -1])
    features = tf.stack([features[:, i:i+19*26] for i in range(0,features.shape[1]-19*26+1,26)],1)
    features = tf.reshape(features, [batch_size, -1, 19*26])

    # 3. Whiten the data
    mean, var = tf.nn.moments(features, axes=[0,1,2])
    features = (features-mean)/(var**.5)

    # 4. Finally we process it with DeepSpeech
    logits = DeepSpeech.BiRNN(features, length, [0]*10)

    return logits

