import json

data = json.load(open('/home/u2/git/hypohyper/output/mwe/TESTfreq_araneum-rncwiki-news-rncP-pro_ruthesNOUNs.json'))

for k,v in data.items():
    print(k,v)
    
print(len(data))