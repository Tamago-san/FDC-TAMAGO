
import pandas as pd
import numpy as np
import ctypes
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore



OUT_NODE=1
SPECTRAL_RADIUS = 1
ITRMAX=5000000
Traning_Ratio = 100
R0= 1.
GOD_STEP=1
RC_NODE=2000
steps_of_history=0
target_num=25
RC_STEP=20
PLOT_num=7


def call_create_dataset01(df01,df02,df03,df04):
#def call_create_dataset(df3,df4):
    df01=df01.rename(columns = str.upper)
    df02=df02.rename(columns = str.upper)
    df03=df03.rename(columns = str.upper)
    df04=df04.rename(columns = str.upper)
    df= pd.concat([df01,df02,df03,df04],axis=0)
#    df= pd.concat([df3,df4],axis=0)
    df["VALUE_DATE"] = pd.to_datetime(df["VALUE_DATE"])
    df=df.sort_values(by=[ "STOCK_CODE","VALUE_DATE"], ascending=True)
    df1 = df.set_index(["STOCK_CODE","VALUE_DATE"], drop=True)
    print(df1)
    print(df1.sum(level=0))
    df2=df1.sum(level=0)
    df2=df2.sort_values(by=["VOLUME"], ascending=False)
    print(df2)
    name_index = list(df2.index[0:target_num])
    print(name_index)

    df4=pd.DataFrame()
    for index in name_index:
        df3=df1.loc[index]
        df3=df3.rename(columns={"CLOSE":index})
        df3=df3.drop("VOLUME", axis=1)
        df4=pd.concat([df4,df3],axis=1)
    
    print(df4)
    data=df4.astype("float64").values
    data = zscore(data,axis=0)
    
    return  data,name_index
    
    
def call_create_dataset02(data):

    len_index=data.shape[0]
    len_column =data.shape[1]
#    print(len(data.index))
    X,ORIGINAL,FUTURE = [],[],[]
    X_tmp=np.empty((len_index-steps_of_history-GOD_STEP,len_column*steps_of_history))
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        for xno in range(0,len_column):
            X_tmp[i,xno*steps_of_history:xno*steps_of_history+steps_of_history] = data[i:i+steps_of_history,xno]
    
    for i in range(0, len_index-steps_of_history-GOD_STEP):
        ORIGINAL.append(data[i+steps_of_history,0:len_column])
        FUTURE.append(data[i+GOD_STEP+steps_of_history,0])
    
    X=X_tmp
    print(np.array(X).shape)
    print(np.array(ORIGINAL).shape)
    print(np.array(FUTURE).shape)
    X = np.reshape(np.array(X), [len_index-steps_of_history-GOD_STEP,steps_of_history*len_column])
    ORIGINAL = np.reshape(np.array(ORIGINAL), [len_index-steps_of_history-GOD_STEP,len_column])
    FUTURE = np.reshape(np.array(FUTURE), [len_index-steps_of_history-GOD_STEP,OUT_NODE])
    print(X.shape)
    print(ORIGINAL.shape)
    print(FUTURE.shape)
    dataset=np.hstack((X,ORIGINAL,FUTURE))
    print(dataset.shape)
    print(dataset)
    np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
    return dataset
    
def call_fortran_rc_karman(_in_node,_out_node,_rc_node,_traning_step,_rc_step,
                        U_in,S_out,U_rc,S_rc,W_out):
    f = np.ctypeslib.load_library("rc_karman.so", ".")
    f. rc_traning_own_karman_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. rc_traning_own_karman_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))

    f.rc_traning_own_karman_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
                            U_in,S_out,U_rc,S_rc,W_out)
                            
print("call read csv 1")
dataframe1 = pd.read_csv('./data/stock_2019.csv',
        usecols=[0,1,3,6],
        engine='python',
        skiprows=0,
        skipfooter=0)
print("call read csv 2")
dataframe2 = pd.read_csv('./data/stock_2018.csv',
        usecols=[0,1,3,6],
        engine='python',
        skiprows=0,
        skipfooter=0)
print("call read csv 3")
dataframe3 = pd.read_csv('./data/stock_2017.csv',
        usecols=[0,1,3,6],
        engine='python',
        skiprows=0,
        skipfooter=0)
print("call read csv 4")
dataframe4 = pd.read_csv('./data/stock_2016.csv',
        usecols=[0,1,3,6],
        engine='python',
        skiprows=0,
        skipfooter=0)

print("call create data")
DATASET, name_list = call_create_dataset01(dataframe1,dataframe2,dataframe3,dataframe4)

DATA_rc = np.zeros((RC_STEP,0))
for i in range(0,target_num):
    DATA_tmp0 = DATASET[:,i].reshape(-1,1)
    DATA_tmp1 = call_create_dataset02(DATA_tmp0)
    print(DATASET)
    datalen0 = DATA_tmp1.shape[0] #時間長さ
    datalen1 = DATA_tmp1.shape[1] #入力＋出力長さ
    TRANING_STEP = int(datalen0*Traning_Ratio/100)
#    RC_STEP = datalen0 - TRANING_STEP #トレーニングとRCとを分ける
    IN_NODE = datalen1 - OUT_NODE
    W_out = np.empty((RC_NODE,OUT_NODE))
    r_befor = np.zeros((RC_NODE))
    S_rc = np.zeros((RC_STEP,OUT_NODE))
    U_rc = np.zeros((RC_STEP,OUT_NODE))
    #+++++++++++++++++++++++++++++++++++++++++++++++++++
    #===================================================
    #===================================================
    
    U_in  = DATA_tmp1[:,0:IN_NODE]
    S_out = DATA_tmp1[:,IN_NODE:IN_NODE+OUT_NODE]
#    U_in = zscore(U_in,axis=0)
#    S_out = zscore(S_out,axis=0)
    
    print(U_in.shape)
    print(S_out.shape)
    
    #call_fortran_rc_traning_own(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP,GUSAI,ALPHA,G
    #            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
    #            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
    #            ,W_out)
    
    call_fortran_rc_karman(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP
                ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
                ,U_rc[0:RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
                ,W_out)
    
    print(S_out.shape)
#    DATA_rc=np.hstack((DATA_rc,S_rc))
    DATA_rc = np.r_["1",DATA_rc,S_rc.reshape(-1,1)]
    
OUTPUT_FILE=pd.DataFrame(DATA_rc,columns=name_list)
OUTPUT_FILE=OUTPUT_FILE.sort_values(RC_STEP-1, axis=1, ascending=False)
#OUTPUT_FILE=OUTPUT_FILE.drop(8918, axis=1)
OUTPUT_FILE.to_csv('./data_out/out_rc.csv')



print(OUTPUT_FILE)

#2次元プロット
#for  list in name_list:
#    plt.plot(DATA_rc[0:RC_STEP,name_list.index(list)],"-" , label=list)

OUTPUT_FILE.plot(y=OUTPUT_FILE.columns[0:PLOT_num])

#plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.show()


    