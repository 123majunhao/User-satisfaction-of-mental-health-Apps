import pandas as pd
import urllib3
import json
import time
import numpy

data=pd.read_csv('E:\qinggan.csv',encoding='ansi')
review_list=data.review

labels=[]
label_prediction =[]

access_token='24.f0da1c09cc03127565e4731bf93accaf.2592000.1695613966.282335-37845346'
http=urllib3.PoolManager()
url='https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token='+access_token

for i in range(len(review_list)):
    if (i + 1) % 2 == 0:
        time.sleep(1)  
    params={'text':review_list[i]}
    encoded_data = json.dumps(params).encode('utf-8')
    request = http.request('POST',
                           url,
                           body=encoded_data,
                           headers={'Content-Type': 'application/json'})
    result = str(request.data, 'utf-8')
    a = json.loads(result)
    output = a['items'][0]
    labels.append(output['sentiment'])
    label_prediction.append(output['positive_prob'])
    print("第 " + str(i + 1) + "条评论已完成分析")
    
data['label']=labels   
data['positive_prob']=label_prediction 
data.to_csv('D:\qingganjieguo.csv',encoding="utf-8")
