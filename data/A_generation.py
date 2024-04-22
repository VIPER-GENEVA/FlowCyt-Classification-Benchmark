import FlowCal
import pandas as pd
import os

p = dict()
for i in range(1, 31):
    p[i-1] = dict()
    for ch in ['O', 'N' , 'G', 'P', 'K', 'B']:
        p[i-1][ch] = FlowCal.io.FCSData(f'data/raw/Case{i}_' + ch + '.fcs')
        
# Dataset generation for total population classification
column=('FS INT', 'SS PEAK', 'SS INT', 'SS TOF', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO', 'TIME')
for i in range(30):
    df_O=pd.DataFrame(p[i]['O'],columns=column)
    df_N=pd.DataFrame(p[i]['N'],columns=column)
    df_G=pd.DataFrame(p[i]['G'],columns=column)
    df_P=pd.DataFrame(p[i]['P'],columns=column)
    df_K=pd.DataFrame(p[i]['K'],columns=column)
    df_B=pd.DataFrame(p[i]['B'],columns=column)
    
    df_O['label']=0 # T Lymphocytes
    df_N['label']=1 # B Lymphocytes
    df_G['label']=2 # Monocytes
    df_P['label']=3 # Mast cells
    df_K['label']=4 # HSPCs
    df_B['label']=5 # Others

    directory = "data/data_original"
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.concat([df_O,df_N,df_G,df_P,df_K,df_B])
    df.to_csv(f"{directory}/Case_{i+1}.csv",index=False)
