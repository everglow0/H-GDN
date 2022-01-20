import torch
import numpy as np
from sklearn.cluster import SpectralClustering

from main import Main

n_clusters = 4
cos_values = torch.load("/home/gpu/guoht/GDN/pretrained/cos_values.pth").cpu()
index = torch.load("/home/gpu/guoht/GDN/pretrained/index.pth").cpu()
embedding = torch.load("/home/gpu/guoht/GDN/pretrained/embedding.pth").cpu()

# np.savetxt('/home/gpu/guoht/GDN/pretrained/embedding.tsv', embedding, delimiter='\t')
embedding = embedding.numpy()
# cluster = AgglomerativeClustering(n_clusters=4, affinity="cosine", linkage="complete").fit(embedding).labels_ # pass
# cluster = SpectralBiclustering(n_clusters=4, random_state=5).fit(embedding).row_labels_ # pass
# cluster = AffinityPropagation(random_state=5).fit(embedding).labels_ # 类的数量无法指定
cluster = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=5).fit(embedding).labels_
# cluster = KMeans(n_clusters=4, random_state=0).fit(embedding).labels_
print(cluster)
ind = np.array([x for x in range(0,27)])
conc = np.concatenate((cluster, ind))
conc = conc.reshape(2, -1).T
conc = np.insert(conc, 0, values=np.array([222,111]), axis=0)
np.savetxt('/home/gpu/guoht/GDN/pretrained/label.tsv', conc, delimiter='\t')
# np.savetxt('/home/gpu/guoht/GDN/pretrained/embedding.tsv', cls.labels_, delimiter='\t')