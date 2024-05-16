import pandas as pd

def check_au_presence(row, au_columns):
    present_aus = []
    for au in au_columns:
        if row[au] == 1: present_aus.append(au)
    return present_aus

def top5_au(path):
    dic = {' AU04_c': 'Brow Lowerer',
           ' AU12_c': 'Lip Corner Puller',
           ' AU14_c': 'Dimpler',
           ' AU10_c': 'Upper Lip Raiser',
           ' AU05_c': 'Upper Lid Raiser',
           ' AU07_c': 'Lid Tightener',
           ' AU06_c': 'Cheek Raiser',
           ' AU23_c': 'Lip Tightener',
           ' AU17_c': 'Chin Raiser',
           ' AU15_c': 'Lip Corner Depressor',
           ' AU45_c': 'Blink',
           ' AU02_c': 'Outer Brow Raiser',
           ' AU25_c': 'Lips part',
           ' AU20_c': 'Lip stretcher',
           ' AU01_c': 'Inner Brow Raiser',
           ' AU26_c': 'Jaw Drop',
           ' AU09_c': 'Nose Wrinkler',
           ' AU28_c': 'Lip Suck'}

    au_columns = [' AU04_c', ' AU12_c', ' AU14_c', ' AU10_c', ' AU05_c', ' AU07_c', ' AU06_c', ' AU23_c', ' AU17_c',
                  ' AU15_c', ' AU45_c', ' AU02_c', ' AU25_c', ' AU20_c', ' AU01_c', ' AU26_c', ' AU09_c', ' AU28_c']
    au_counts = {au: 0 for au in au_columns}

    data = pd.read_csv(path)
    for index, row in data.iterrows():
        present_aus = check_au_presence(row, au_columns)
        for au in present_aus:
            au_counts[au] += 1

    sorted_au_counts = sorted(au_counts.items(), key=lambda item: item[1], reverse=True)
    top5_aus = [dic[au] for au, count in sorted_au_counts[:5]]

    return top5_aus


if __name__ == "__main__":
    print(top5_au('D:/Monash/semester 6/FIT 3162/Mexp_Features/Mexp_reallifedeception_trial_lie_044.csv'))
