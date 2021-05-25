# -*- coding: utf-8 -*- 
import MeCab, csv, os
import glob, pandas as pd, numpy as np
m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
os.chdir("/Users/noyeongdan/영화 리뷰")
root_dir = "/Users/noyeongdan/영화 리뷰/data"
category_, id_, content_ = ['label'], ['id'], ['doc']
#file_name = ['Article_세계_201901_201901']
#file_name = ['Article_생활문화_201901_201901','Article_세계_201901_201901','Article_IT과학_201901_201901']
file_name = ['ratings2']

def file_to_ids(fname):
    #name = fname.split(".csv")[0]
    #name_array = name.split('/')
    #f_name = name_array[3] #파일이름
    
    #tags for tokenizer
    #tag_classes = ['XSV+EC']
    tag_classes = ['NNG', 'NNP','VA', 'VV+EC', 'VV+ETM', 'MAJ', 'XR', 'VA+EC']
    #데이터 읽어오고.
    data = pd.read_csv(fname+'.csv')
    #각각 분류
    id = data.iloc[:,0].values #title
    doc = data.iloc[:, 1].values
    label = data.iloc[:, 2].values #content

    for cnt, value in enumerate(id):
        result = ''
        value = m.parseToNode(str(doc[cnt]).strip())
        while value:
            tag = value.feature.split(",")[0]
            word = value.feature.split(",")[3]
            if tag in tag_classes:
                if word == "*": value = value.next
                result += word.strip()+" "
            value = value.next
        content_.append(result)
        id_.append(id[cnt])
        
        #category
        category_.append(str(label[cnt]).strip())
#         #category
#         if '스포츠' in fname : category_.append("0")
#         if '세계' in fname : category_.append("1")
#         if 'IT과학' in fname : category_.append("2")

def save(month, file_path, f_name):
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    with open(file_path+"/"+f_name+'_after_prepro.csv', 'w') as f:
        writer = csv.writer(f)
    
        for cnt, i in enumerate(content_):
            id__ = id_[cnt]
            content__ = content_[cnt]
            category__ = category_[cnt]
            writer.writerow((id__, content__, category__))
#             if f_name=='Article_일반 스포츠_201901_201901' and category_[cnt]=='0':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_세계_201901_201901' and category_[cnt]=='1':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
#             elif f_name=='Article_IT과학_201901_201901' and category_[cnt]=='2':
#                 date__ = date_[cnt]
#                 content__ = content_[cnt]
#                 category__ = category_[cnt]
#                 writer.writerow((date__, content__, category__))
                
                
         #for cnt, i in enumerate(content_):
         #   print(category_[cnt])
         #   if category_[cnt]==0:
         #       print('int')
         #   elif category_[cnt]=='0':
         #       print('string')

for name in file_name:
    file_to_ids(name)
    #register_dic(name)
    save(name, '/Users/noyeongdan/영화 리뷰', name)
#save_files()