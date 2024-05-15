import pandas as pd

data = pd.read_csv('D:/Monash/semester 6/FIT 3162/Mexp_Features/Mexp_reallifedeception_trial_lie_001.csv')
au_columns = [' AU04_c', ' AU12_c', ' AU14_c', ' AU10_c', ' AU05_c', ' AU07_c', ' AU06_c']
dic = {' AU04_c': 'Brow Lowerer',
       ' AU12_c': 'Lip Corner Puller',
       ' AU14_c': 'Dimpler',
       ' AU10_c': 'Upper Lip Raiser',
       ' AU05_c': 'Upper Lid Raiser',
       ' AU07_c': 'Lid Tightener',
       ' AU06_c': 'Cheek Raiser'}

# au_columns = [' AU04_c', ' AU12_c', ' AU14_c', ' AU10_c', ' AU05_c', ' AU07_c', ' AU06_c', ' AU23_c', ' AU17_c',
#               ' AU15_c', ' AU45_c', ' AU02_c', ' AU25_c', ' AU20_c', ' AU01_c', ' AU26_c', ' AU09_c', ' AU28_c']

def check_au_presence(row, au_columns):
    present_aus = []
    for au in au_columns:
        if row[au] == 1:
            present_aus.append(au)
    return present_aus

for index, row in data.iterrows():
    present_aus = check_au_presence(row, au_columns)
    if present_aus:
        au_names = []
        for i in range(len(present_aus)):
            au_names.append(dic[present_aus[i]])
        print(f'Row {index}: Present AUs - {", ".join(au_names)}')
        # print(f'Row {index}: Present AUs - {", ".join(present_aus)}')
