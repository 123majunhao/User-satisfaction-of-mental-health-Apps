import pandas as pd
import json
import urllib.request
import urllib.parse
import urllib.request


#data=pd.read_csv('E:\qinggan.csv',encoding='utf-8')
#review_list=data.review

url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"
values = {
 'host':'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials',
 'client_id':'VMojOedoG6N5TtB7PdZtxYBG',  
 'client_secret' : 'vw9szPpet7O5zTXOjvrdyCSLh8Uu4zhC' 
}

host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=VMojOedoG6N5TtB7PdZtxYBG&client_secret=vw9szPpet7O5zTXOjvrdyCSLh8Uu4zhC'
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
if (content):
    print(content)  
