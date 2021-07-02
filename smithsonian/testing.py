import pickle 
from glob import glob 

data_order = pickle.load(open('data_order.pickle', 'rb'))
# print(data_order[0:10])

# for val in data_order[0:20]:
#     split_str = val.split('\\')[-1].split('.jpg')[0]
#     print(val, split_str)

from sklearn.metrics.pairwise import cosine_similarity

files = glob('encoded_imgs/*.pickle')
number_of_arrs = []
for i, f in enumerate(files):
    arr = pickle.load(open(f, 'rb'))

    print(cosine_similarity(
        arr['edanmdm-saam_1906.9.15'].reshape(1, -1),
        arr['edanmdm-saam_1882.1.1'].reshape(1, -1)
    ))

    break



#     number_of_arrs.append(arr.shape[0])

# print(len(data_order))
# print(sum(number_of_arrs))



