import requests

url = 'http://localhost:5000/classificationAPI'

result = requests.post(url,json={'age':'36', 
                                 'height':'175', 
                                 'weight':'66', 
                                 'gender':'1', 
                                 'sys':'110', 
                                 'dias':'80', 
                                 'chol':'3', 
                                 'glu':'3',
                                 'alc':'0', 
                                 'smo':'0', 
                                 'phyact':'0' 
                                 })

print(result.json())