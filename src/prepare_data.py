from datasets import load_dataset
import argparse
import numpy as np
from nltk.stem import PorterStemmer
import json

def split_texts(dataset):
    dataset["stemm_title"] = dataset["title"].split(" ")
    dataset["stemm_abstract"] = dataset["abstract"].split(" ")
    return dataset

def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))

"""
On split la source par un espace.
Pour chaque élément de la source split, si sa version lower contient le MC
alors on replace par le marqueur et src[i] est remplacé par lowered[i]
"""
def drop_kp(dataset, p=0.7):
    stemmer = PorterStemmer()
    artificial_instances = {}

    for k,element in enumerate(dataset):
        presents = np.where(np.isin(element["prmu"], "P"))[0].tolist()
        present_keyphrases = [element["keyphrases"][i] for i in presents]
        tokenized_kps = [kp.split(" ") for kp in present_keyphrases]
        stemmed_kps = []
        for tokenized_kp in tokenized_kps:
            stemmed_kp = " ".join([stemmer.stem(kw.lower().strip()) for kw in tokenized_kp])
            stemmed_kps.append(stemmed_kp)

        kps_drop = np.random.binomial(1, p, len(stemmed_kps))
        kps_drop = kps_drop.tolist()

        new_presents = [present_keyphrases[i] for i in range(len(present_keyphrases)) if kps_drop[i] == 0]
        new_absents = [present_keyphrases[i] for i in range(len(present_keyphrases)) if kps_drop[i] == 0]
        absents = np.where(np.isin(element["prmu"], "P",invert=True))[0].tolist() # Get all the keyphrases that are NOT a present keyphrase
        og_absent_keyphrases = [element["keyphrases"][i] for i in absents]

        new_keyphrases = [*new_presents,*new_absents,*og_absent_keyphrases]


        if kps_drop.count(1) > 0: #if there are keyphrases to mask

            title = element["title"]
            abstract = element["abstract"]

            splitted_title = title.split(" ")
            splitted_abstract = abstract.split(" ")

            lowered_title = " ".join([stemmer.stem(token.lower().strip()) for token in splitted_title])
            
            lowered_abstract = " ".join([stemmer.stem(token.lower().strip()) for token in splitted_abstract])

            kps_to_drop = [stemmed_kps[i] for i in np.where(np.isin(kps_drop, 1))[0].tolist()]
            
            for kp in kps_to_drop: # remplace les kp à drop par # 
                lowered_title = lowered_title.replace(kp," ".join(["#"*len(kp.split(" "))]))
                lowered_abstract = lowered_abstract.replace(kp," ".join(["#"*len(kp.split(" "))]))

            new_title = []
            for og_token,token in zip(splitted_title,lowered_title.split(" ")):
                if "#" not in token:
                    new_title.append(og_token)
                elif new_title:
                    if new_title[-1] != "<pad>":
                        new_title.append("<pad>")
                else:
                    new_title.append("<pad>")
            
            new_abstract = []
            for og_token,token in zip(splitted_abstract,lowered_abstract.split(" ")):
                if "#" not in token:
                    new_abstract.append(og_token)
                elif new_abstract:
                    if new_abstract[-1] != "<pad>":
                        new_abstract.append("<pad>")
                else:
                    new_abstract.append("<pad>")


            # new_title=[]
            # for i,token in enumerate(lowered_title):
            #     for j, kp in enumerate(stemmed_kps):
            #         if contains(kp,token):
            #             if kps_drop[j]==1:
            #                 new_title.append(token.replace(kp,"<pad>"))
                            
            #             else:
            #                 new_title.append(splitted_title[i]) # We keep the original token
            #             break
            #         else:
            #             new_title.append(splitted_title[i]) # We keep the original token
            #             break

            
            # new_abstract=[]
            # for i,token in enumerate(lowered_abstract):
            #     for j, kp in enumerate(stemmed_kps):
            #         if contains(kp,token):
            #             if kps_drop[j]==1:
            #                 new_abstract.append(token.replace(kp,"<pad>"))
                            
            #             else:
            #                 new_abstract.append(splitted_abstract[i]) # We keep the original token
            #             break
            #         else:
            #             new_abstract.append(splitted_abstract[i]) # We keep the original token
            #             break

                artificial_instances[element["id"]] = {}
                artificial_instances[element["id"]]["title"] = " ".join(new_title)
                artificial_instances[element["id"]]["abstract"] = " ".join(new_abstract)
                artificial_instances[element["id"]]["keyphrases"] = new_keyphrases

        if k%1000==0:
            print(k)

    return artificial_instances

if __name__ == "__main__":
    
    np.random.seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--data")

    parser.add_argument("-o","--output_file")

    args = parser.parse_args()

    data = load_dataset("json",data_files=args.data)

    artificial_instances = drop_kp(data["train"])

    with open(args.output_file,"a") as f:
        for key in artificial_instances.keys():
            json.dump(artificial_instances[key],f)
            f.write("\n")
