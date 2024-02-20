import os



# 图片id-类别id
image_class_labels = "./AwA2/AwA2-labels.txt"
with open(image_class_labels, 'r') as f:
    image_class_labels_lines =  f.readlines()

# 属性维度-文本描述
#attr_id2text = {}
attributes = "./AwA2/Animals_with_Attributes2/predicates.txt"
attr_names = []
with open(attributes, 'r') as f:
    attributes_lines =  f.readlines()
    for line in attributes_lines:
        attr_id, attr_text = line.strip().split("\t")
        #attr_id2text[attr_id] = attr_text
        attr_names.append(attr_text)
print(attr_names)

# 每个类别对应的属性值 
class_attr = "./AwA2/Animals_with_Attributes2/predicate-matrix-binary.txt"
class_attr_dict = {}  #0-49  每个类别为1的属性
with open(class_attr, 'r') as f:
    class_attr_lines =  f.readlines()
    for i,line in enumerate(class_attr_lines):
        # print(line)
        attrs = line.strip().split(" ")
        #print(len(attrs))
        assert len(attrs)==85
        idx = []
        for j,v in enumerate(attrs):
            if int(v)>0:
                idx.append(j)
        class_attr_dict[i] = idx 


print("total imgs: ", len(image_class_labels_lines))
final_text = []
for i,line in enumerate(image_class_labels_lines):
    img_id = i+1
    class_id = int(line.strip())-1
 
    attr_items = class_attr_dict[class_id]
    #print(len(attr_items), attr_items[:2])
    
    full_sentense = "The animal on image id %s has these attributes: " % (img_id)
    attr_sentense = ""

    for attr_item in attr_items:
        attr_name = attr_names[attr_item]
       

        attr_text = "%s, " % (attr_name)
        #print(attr_text)
        attr_sentense+=attr_text

    
    final_text.append(full_sentense +attr_sentense[:-2]+".")
    #break

with open("awa2_attr_text.txt", 'w') as f:
    for text in final_text:
        f.write(text+"\n")


