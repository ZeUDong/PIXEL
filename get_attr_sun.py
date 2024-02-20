import os
import numpy as np


# 图片id-属性值
img_attrs  = np.load("./SUN/SUNAttributeDB/attr.npy")


# 属性维度-文本描述

attributes = "./SUN/SUNAttributeDB/attr_names.txt"
attr_names = []
with open(attributes, 'r') as f:
    attributes_lines =  f.readlines()
    for line in attributes_lines:
        attr_name = line.strip()
        attr_names.append(attr_name)

# 属性的确定性
"""
1 not visible
2 guessing
3 probably
4 definitely
"""
print("total imgs: ", len(img_attrs))

final_text = []
for img_id,data in enumerate(img_attrs):


    full_sentense = "The scene on image id %s has these attributes: " % (img_id)
    attr_sentense = ""

    for j,attr_v in enumerate(data):
        if attr_v==0:
            continue
        elif attr_v<0.5:
            cert_word = "probably"
        elif attr_v<0.7:
            cert_word = "probably"
        else:
            cert_word = "definitely"

        attr_text = cert_word+" "+attr_names[j]+", "
        attr_sentense+=attr_text

    final_text.append(full_sentense +attr_sentense[:-2]+".")
    #break

with open("sun_attr_text.txt", 'w') as f:
    for text in final_text:
        f.write(text+"\n")


