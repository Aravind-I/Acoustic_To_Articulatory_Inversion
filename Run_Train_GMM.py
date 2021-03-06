#!/usr/bin/env python
# coding: utf-8

# In[1]:
#https://r9y9.github.io/nnmnkwii/v0.0.1/nnmnkwii_gallery/notebooks/tts/01-DNN-based%20statistical%20speech%20synthesis%20(en).html

import numpy as np
import scipy.io
from scipy.io import wavfile
from sklearn.mixture import GaussianMixture
import librosa
import librosa.display
import os
import HTK
from scipy.stats import pearsonr
from keras.preprocessing.sequence import pad_sequences
from nnmnkwii.preprocessing import delta_features
import pickle
#from nnmnkwii.baseline.gmm import MLPG
from nnmnkwii.paramgen import mlpg
from Utils_GMM import GMM_M
from KalmanSmoother import *
from scipy.signal import butter, filtfilt
from scipy import signal
from nnmnkwii.util import trim_zeros_frames, remove_zeros_frames
# In[2]:

fb, fa = signal.cheby2(10,40,7.0/(100/2),'low', analog=False)#, 'low', analog=True)#(4, 5, 100, 'low', analog=True)

def Get_Wav_EMA_PerFile(EMA_file,Wav_file,F):
    EmaMat=scipy.io.loadmat(EmaDir+EMA_file);
    EMA_temp=EmaMat['EmaData'];
    EMA_temp=np.transpose(EMA_temp)# time X 18
    Ema_temp2=np.delete(EMA_temp, [4,5,6,7,10,11],1) # time X 12
    MeanOfData=np.mean(Ema_temp2,axis=0) 
    Ema_temp2-=MeanOfData
    C=0.5*np.sqrt(np.mean(np.square(Ema_temp2),axis=0))
    Ema=np.divide(Ema_temp2,C) # Mean remov & var normailized
    [aE,bE]=Ema.shape
    
    #print F.type
    EBegin=np.int(BeginEnd[0,F]*100)
    EEnd=np.int(BeginEnd[1,F]*100)
    
    ### MFCC ###
    htkfile.load(MFCCpath+Wav_file[:-4]+'.mfc')
    feats = np.asarray(htkfile.data)
    mean_G = np.mean(feats, axis=0)
    std_G = np.std(feats, axis=0)
    feats = std_frac*(feats-mean_G)/std_G
    MFCC_G = feats
    TimeStepsTrack=EEnd-EBegin
    return Ema[EBegin:EEnd,:], MFCC_G[EBegin:EEnd,:n_mfcc],TimeStepsTrack # with out silence

def EvalMetric(X,Y):
    CC=[pearsonr(X[:,i],Y[:,i]) for i in range(0,12)]
    rMSE=np.sqrt(np.mean(np.square(X[:,0:12]-Y[:,0:12]),axis=0))
    return np.array(CC)[:,0],rMSE


windows = [(0, 0, np.array([1.0])),            # static
           (1, 1, np.array([-0.5, 0.0, 0.5])), # delta
         (1, 1, np.array([1.0, -2.0, 1.0])),]


# In[3]:


len(windows)


# In[4]:


htkfile = HTK.HTKFile()
std_frac=0.25
n_mfcc=13

TrainMsub=[ 'AshwinHebbar', 'VigneshM', 'Advith', 'NikhilB', 'Pavan_P', 'AdvaithP', 'Shoubik', 'PhaniKumar', 'Prakhar_G','Parth_S']
TrainFsub=[ 'Babitha', 'DivyaGR','Vidhi', 'Anwesa', 'Nisha', 'Chandana','Varshini','Rupasi','Shiny','AtreyeeS']
ValMsub=['Ashwin_N', 'AvinashKumar']
ValFsub=['Tanaya']

RemainingSubs=['RaviKiran_R', 'Samik', 'Pavan', 'ManuG', 'Vignesh', 'GokulS']

ExcludeSet1=['Ankur', 'Anand_S' , 'Prasad','Ankur_C','Monika', 'SriRamya','Harshini','Jisha','Anisha']


Valsubs=ValMsub+ValFsub
Trainsubs=TrainMsub+TrainFsub


# In[18]:
TSCC=np.zeros((1,12*2))
TSrmse=np.zeros((1,12*2))

TSCC_MLPG=np.zeros((1,12*2))
TSrmse_MLPG=np.zeros((1,12*2))

TSCC_KM=np.zeros((1,12*2))
TSrmse_KM=np.zeros((1,12*2))

TSCC_LPF=np.zeros((1,12*2))
TSrmse_LPF=np.zeros((1,12*2))

OutDir='GMM_AAI_Model/'
RootDir='../../../SPIRE_EMA/'
for ss in np.arange(0,len(Trainsubs)):

    X_valseq=[];Youtval=[];
    X_trainseq=[];Youttrain=[];
    X_testseq=[];Youttest=[];
    TT_Test=[];TT_Train=[];TT_Valid=[]
    print('Loading Training data')
#for ss in np.arange(0,len(Trainsubs)):
    Sub=Trainsubs[ss]#'Anand_S'
    print(Sub)
    WavDir=RootDir+'DataBase/'+Sub+'/Neutral/WavClean/';
    EmaDir=RootDir+'DataBase/'+Sub+'/Neutral/EmaClean/';
    BeginEndDir=RootDir+'/StartStopMat/'+Sub+'/';
    MFCCpath=RootDir+'/DataBase/'+Sub+'/Neutral/MfccHTK/'

    EMAfiles=sorted(os.listdir(EmaDir))
    Wavfiles=sorted(os.listdir(WavDir))
    StartStopFile=os.listdir(BeginEndDir)
    StartStopMAt=scipy.io.loadmat(BeginEndDir+StartStopFile[0])
    BeginEnd=StartStopMAt['BGEN']
    #window_size=500


    F=5 # Fold No
    '''
    for i in np.arange(0,460):
        E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
        #W_t=W_t#[np.newaxis,:,:,:]
        E_t=E_t#[np.newaxis,:,:]
        M_t=M_t#[np.newaxis,:,:]
        Youttrain.append(E_t)
        X_trainseq.append(M_t)
        TT_Train.append(TT)
    '''
    for i in np.arange(0,460):
        if  (((i+F)%10)==0):# Test
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            E_t= delta_features(E_t, windows)
            M_t= delta_features(M_t, windows)
            #W_t=W_t[np.newaxis,:,:,np.newaxis]
            E_t=E_t[np.newaxis,:,:]
            M_t=M_t[np.newaxis,:,:]
            Youttest.append(E_t)
            X_testseq.append(M_t)
            TT_Test.append(TT)
            #print('Test '+str(i))
        elif (((i+F+1)%10)==0):# Validation
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            E_t= delta_features(E_t, windows)
            M_t= delta_features(M_t, windows)
            #W_t=W_t#[np.newaxis,:,:,:]
            E_t=E_t#[np.newaxis,:,:]
            M_t=M_t #forward_process(M_t[:,:n_mcep])
            Youtval.append(E_t)
            X_valseq.append(M_t)
            TT_Valid.append(TT)
        else: # Train
            E_t,M_t,TT=Get_Wav_EMA_PerFile(EMAfiles[i],Wavfiles[i],i)
            E_t= delta_features(E_t, windows)
            M_t= delta_features(M_t, windows)
            #W_t=W_t#[np.newaxis,:,:,:]
            E_t=E_t#[np.newaxis,:,:]
            M_t=M_t #forward_process(M_t[:,:n_mcep])#[np.newaxis,:,:]
            Youttrain.append(E_t)
            X_trainseq.append(M_t)
            TT_Train.append(TT)
    # In[23]:


    print("No:of Train Samples:"+str(len(X_trainseq)))
    print("No:of Val Samples:"+str(len(X_valseq)))


    # In[5]:


    TT_Total=(np.concatenate([np.array(TT_Test),np.array(TT_Valid),np.array(TT_Train)]))
    TT_max=400#np.max(TT_Total)
    #padded = pad_sequences(sequences, padding='post',maxlen=TT_max)
    X_valseq=pad_sequences(X_valseq, padding='post',maxlen=TT_max,dtype='float')
    Youtval=pad_sequences(Youtval, padding='post',maxlen=TT_max,dtype='float')
    X_trainseq=pad_sequences(X_trainseq, padding='post',maxlen=TT_max,dtype='float')
    Youttrain=pad_sequences(Youttrain, padding='post',maxlen=TT_max,dtype='float')
    #X_testseq=pad_sequences(X_testseq, padding='post',maxlen=TT_max)
    #Youttest=pad_sequences(Youttest, padding='post',maxlen=TT_max)



    XY=np.concatenate((X_trainseq,Youttrain),axis=-1).reshape(-1, X_trainseq.shape[-1]+Youttrain.shape[-1])
    xy = remove_zeros_frames(XY)


    filename =OutDir+Sub+'32_full_finalized_model.sav'

    gmm = GaussianMixture(n_components=32, covariance_type="full", max_iter=1000, verbose=1)
    gmm.fit(xy)
    #%time gmm.fit(xy) #XY
    pickle.dump(gmm, open(filename, 'wb'))
    #=====Loading Saved Model =======
    #filename ='finalized_model.sav'
    #gmm= pickle.load(open(filename, 'rb'))


    # In[32]:


    print('====Testing=====')
    print("No:of Train Samples:"+str(len(X_testseq)))
    predSeq_wom=np.empty((1,len(Youttest)), dtype=object);#With out MLPG
    predSeq_wm=np.empty((1,len(Youttest)), dtype=object);#With MLPG
    predSeq_kf=np.empty((1,len(Youttest)), dtype=object);#With KalmanFilter
    predSeq_lpf=np.empty((1,len(Youttest)), dtype=object);#With LPF
    YtestOrg=np.empty((1,len(Youttest)), dtype=object);
    tCC=[]
    trMSE=[]
    tCC_MLPG=[]
    trMSE_MLPG=[]
    tCC_KM=[]
    trMSE_KM=[]
    tCC_LPF=[]
    trMSE_LPF=[]
    paramgen = GMM_M(gmm, windows=windows) # Inherit the GMM class
    for i in np.arange(0,len(Youttest)):
        s_in=np.squeeze(X_testseq[i])
        #s_in=s_in[np.newaxis,:,0:inputDim]
        #val=model.predict(s_in);
        val, D, W=paramgen.transform(s_in) # val: Conditional Expectation; D: Conditional Variance; W: windows 
        predSeq_wom[0,i]=val
        predSeq_wm[0,i]=mlpg(val, D, W) #MLPG
        k_smth = kalmansmooth(val.transpose()).transpose() # Kalaman Filtering
        predSeq_kf[0,i]=k_smth
        #InSeq[0,i]=s_in
        yLPF = filtfilt(fb, fa, val.transpose()).transpose()
        predSeq_lpf[0,i]=yLPF        
        
        YtestOrg[0,i]=Youttest[i]

        iCC,irMSE=EvalMetric(val,np.squeeze(Youttest[i]))
        tCC.append(iCC)
        trMSE.append(irMSE)

        iCC,irMSE=EvalMetric(mlpg(val, D, W),np.squeeze(Youttest[i]))
        tCC_MLPG.append(iCC)
        trMSE_MLPG.append(irMSE)  

        iCC,irMSE=EvalMetric(k_smth,np.squeeze(Youttest[i]))
        tCC_KM.append(iCC)
        trMSE_KM.append(irMSE)  

        iCC,irMSE=EvalMetric(k_smth,np.squeeze(Youttest[i]))
        tCC_KM.append(iCC)
        trMSE_KM.append(irMSE) 

        iCC,irMSE=EvalMetric(yLPF,np.squeeze(Youttest[i]))
        tCC_LPF.append(iCC)
        trMSE_LPF.append(irMSE) 
        
    print(np.mean(np.array(tCC),axis=0))
    print(np.mean(np.array(trMSE),axis=0))
    print("MLPG")
    print(np.mean(np.array(tCC_MLPG),axis=0))
    print(np.mean(np.array(trMSE_MLPG),axis=0))
    print("Kalman")
    print(np.mean(np.array(tCC_KM),axis=0))
    print(np.mean(np.array(trMSE_KM),axis=0))
    print("LPF")
    print(np.mean(np.array(tCC_LPF),axis=0))
    print(np.mean(np.array(trMSE_LPF),axis=0))
#     TSrmse=np.concatenate((TSrmse,np.mean(np.array(trMSE),axis=0)[np.newaxis,:]),axis=0)
#     TSCC=np.concatenate((TSCC,np.mean(np.array(tCC),axis=0)[np.newaxis,:]),axis=0)

#     TSrmse=np.concatenate((TSrmse,np.mean(np.array(trMSE_MLPG),axis=0)[np.newaxis,:]),axis=0)
#     TSCC=np.concatenate((TSCC,np.mean(np.array(tCC_MLPG),axis=0)[np.newaxis,:]),axis=0)

    TSrmse=np.concatenate((TSrmse,np.concatenate([np.mean(np.array(trMSE),axis=0), np.std(np.array(trMSE),axis=0)])[np.newaxis,:]),axis=0) #cocat Mean and std across test sent for each sub
    TSCC=np.concatenate((TSCC,np.concatenate([np.mean(np.array(tCC),axis=0), np.std(np.array(tCC),axis=0)])[np.newaxis,:]),axis=0)

    TSrmse_MLPG=np.concatenate((TSrmse_MLPG,np.concatenate([np.mean(np.array(trMSE_MLPG),axis=0), np.std(np.array(trMSE_MLPG),axis=0)])[np.newaxis,:]),axis=0)
    TSCC_MLPG=np.concatenate((TSCC_MLPG,np.concatenate([np.mean(np.array(tCC_MLPG),axis=0), np.std(np.array(tCC_MLPG),axis=0)])[np.newaxis,:]),axis=0)

    TSrmse_KM=np.concatenate((TSrmse_KM,np.concatenate([np.mean(np.array(trMSE_KM),axis=0), np.std(np.array(trMSE_KM),axis=0)])[np.newaxis,:]),axis=0)
    TSCC_KM=np.concatenate((TSCC_KM,np.concatenate([np.mean(np.array(tCC_KM),axis=0), np.std(np.array(tCC_KM),axis=0)])[np.newaxis,:]),axis=0)

    TSrmse_LPF=np.concatenate((TSrmse_LPF,np.concatenate([np.mean(np.array(trMSE_LPF),axis=0), np.std(np.array(trMSE_LPF),axis=0)])[np.newaxis,:]),axis=0)
    TSCC_LPF=np.concatenate((TSCC_LPF,np.concatenate([np.mean(np.array(tCC_LPF),axis=0), np.std(np.array(tCC_LPF),axis=0)])[np.newaxis,:]),axis=0)

scipy.io.savemat(filename+'_Results.mat',{'RMSE':TSrmse,'Corr':TSCC,'RMSE_MLPG':TSrmse_MLPG,'Corr_MLPG':TSCC_MLPG,'RMSE_KM':TSrmse_KM,'Corr_KM':TSCC_KM,'RMSE_LPF':TSrmse_LPF,'Corr_LPF':TSCC_LPF})
