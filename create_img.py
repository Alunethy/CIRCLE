import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import scanpy

def create_img(datasetX, datasetY, dirc):

    X, y = datasetX, datasetY

    tsne = manifold.TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''vis embedding'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()

    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], 10, y)

    def get_key(dct, value):
        k = [k for k, v in dct.items() if v == value]
        return k

    label = []
    for i in range(len(dirc)):
        label.append(get_key(dirc, i))

    plt.legend(handles=scatter.legend_elements()[0], labels=label, title="classes", bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    f = plt.gcf()
    return f

def line_maker(x1,x2,y1,y2,arg):
    f=plt.figure(figsize=(15,9))
    plt.subplot(1,2,1)
    if arg.objective == "SCCL":
        plt.plot(x1,y1["cluster_loss"],color='red',label="cluster_loss")
        plt.plot(x1,y1["contrastive_loss"],color='green',label="contrastive_loss")
    elif arg.objective == "contrastive" or arg.objective=='ablation':
        plt.plot(x1,y1["contrastive_loss"],color='green',label="contrastive_loss")
    elif arg.objective == "clustering":
        plt.plot(x1,y1["cluster_loss"],color='red',label="cluster_loss")
    elif arg.objective=='para':
        plt.plot(x1,y1["contrastive_loss"],color='black',label="contrastive_loss")
        plt.plot(x1,y1["ins"],color='red',label="ins_conloss")
        plt.plot(x1,y1["clu"],color='green',label="clu_conloss")
    else:
        print("Img occur mistakes!!")
    plt.legend()
    plt.xlabel("training epoch")
    plt.ylabel("Losses")
    plt.title("Losses over epoch")


    plt.subplot(1,2,2)
    plt.plot(x2,y2['ARI'],color='black',label="ARI")
    plt.plot(x2,y2['NMI'],color='red',label="NMI")
    plt.plot(x2,y2['AMI'],color='green',label="AMI")
    plt.legend()
    plt.xlabel("epoch")
    plt.title("Cluster scores over evaluation epoch")
    return f

if __name__ == '__main__':
    digits = datasets.load_digits(n_class=5)
    dirc = {'aa': 1, 'bb': 0, 'cc': 3, 'dd': 4, 'gg': 2}
    f = create_img(digits.data, digits.target, dirc)
    f.savefig("best_embedding.jpg")