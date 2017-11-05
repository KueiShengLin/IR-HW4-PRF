
# coding: utf-8

import os
import math
import copy
import re
import time

DOC_NAME = os.listdir("Document")  # Document file name
QUERY_NAME = os.listdir("Query")  # Query file name

QUERY = []
DOCUMENT = []


class VSM:
    def __init__(self, doc_name, query_name, document, query, rank_amount):
        self.doc_name = doc_name
        self.query_name = query_name
        self.document = document
        self.query = query
        self.rank_amount = rank_amount
        self.doc_freq = {}
        self.ans = []

    def df_measure(self):
        for i in range(0, len(self.document) - 1):
            x = list(self.document[i].keys())
            for j in range(0, len(x) - 1):
                if str(x[j]) in self.doc_freq:
                    self.doc_freq[x[j]] += 1
                else:
                    self.doc_freq[x[j]] = 1

    #   Doc : tf = log Normalization idf = Inverse Frequency, Query : Raw freq
    def tf_idf_LNIF_RF(self):
        N = len(self.doc_name)
        Doc_idf = self.doc_freq.copy()
        x = list(Doc_idf.keys())
        for i in range(0,len(Doc_idf)-1):
            Doc_idf[x[i]] = (math.log10(N/Doc_idf[x[i]])) #idf compute

        Doc_tfidf = copy.deepcopy(self.document)
        for j in range(0,len(Doc_tfidf)-1):
            x = list(self.document[j].keys())
            for i in range(0,len(x)-1):
                #u can add the tf compute on this line
                y = self.document[j][x[i]]
                Doc_tfidf[j][x[i]] = (1+(math.log(y,2)))
                #=====================================
                Doc_tfidf[j][x[i]] = Doc_tfidf[j][x[i]]*Doc_idf[x[i]] #tf*idf

        Q_tfidf = copy.deepcopy(self.query)

        for j in range(0,len(Q_tfidf)-1):

            x = list(self.query[j].keys())
            Max = max(self.query[j].values())

            for i in range(0,len(x)-1):
                #u can add the tf compute there
                #====================================
                if(x[i] in Doc_idf):
                    Q_tfidf[j][x[i]] = self.query[j][x[i]]*Doc_idf[x[i]] #tf*idf
                else:
                    Q_tfidf[j][x[i]] = 0

        return(Doc_tfidf,Q_tfidf)

    def VSMC(self, Doc_tfidf, Q_tfidf):

        Ans_T = []
        for q, que_dic in enumerate(Q_tfidf):
            Sim = []
            for j, doc_dic in enumerate(Doc_tfidf):
                a = 0
                b = 0
                for que_voc in que_dic:
                    if que_dic[que_voc] == 0:
                        continue
                    if que_voc in doc_dic:
                        a += que_dic[que_voc] * doc_dic[que_voc]    # 被除數
                    b += pow(que_dic[que_voc], 2)    # 除數1

                c = sum([pow(doc_dic[doc_voc], 2) for doc_voc in doc_dic])  # 除數2

                Sim.append(a / (math.sqrt(b)*math.sqrt(c)))

            Sim_sort = sorted(Sim, reverse=True)

            Ans = []
            for i in range(0, self.rank_amount):
                Ans.append(self.doc_name[Sim.index(Sim_sort[i])])
            Ans_T.append(Ans)

        return Ans_T

    def writeAns(self, file_name):
        with open(str(file_name) + '.txt', 'w') as file:
            file.write("Query,RetrievedDocuments\n")
            for i in range(0, len(self.query_name)):
                file.write(str(self.query_name[i]) + ',')
                for num, j in enumerate(self.ans[i]):
                    if num < self.rank_amount:
                        file.write(str(j) + ' ')
                    else:
                        break
                file.write('\n')

    def calculate(self):

        self.df_measure()
        print('df down')

        (Doc_tfidf_LNIF_RF, Q_tfidf_LNIF_RF) = self.tf_idf_LNIF_RF()
        print('tfidf down')

        self.ans = self.VSMC(Doc_tfidf_LNIF_RF, Q_tfidf_LNIF_RF)
        print('VSM down')
        # self.writeAns(10)


def readfile():
    global QUERY, DOCUMENT
    global QUERY_NAME, DOC_NAME

    # read document , create dictionary
    for doc_id in DOC_NAME:
        doc_dict = {}
        with open("Document\\" + doc_id) as doc_file:
            doc_file_content = doc_file.read()
            doc_voc = re.split(' |\n', doc_file_content)
            doc_voc = list(filter('-1'.__ne__, doc_voc))
            for dv_id, dv_voc in enumerate(doc_voc):
                if dv_id < 5:
                    continue
                if dv_voc in doc_dict:
                    doc_dict[dv_voc] += 1
                else:
                    doc_dict[dv_voc] = 1
            if '' in doc_dict:  # ? error
                doc_dict.pop('')
        DOCUMENT.append(doc_dict)

    for query_id in QUERY_NAME:
        query_dict = {}
        with open("Query\\" + query_id) as query_file:
            query_file_content = query_file.read()
            query_voc = re.split(' |\n', query_file_content)
            query_voc = list(filter('-1'.__ne__, query_voc))
            for qv_id, qv_voc in enumerate(query_voc):
                if qv_voc in query_dict:
                    query_dict[qv_voc] += 1
                else:
                    query_dict[qv_voc] = 1
            if '' in query_dict:  # ? error
                query_dict.pop('')
        QUERY.append(query_dict)

    print('read file down')


# readfile()
# VSMa = VSM(DOC_NAME, QUERY_NAME, DOCUMENT, QUERY, 5)
# VSMa.calculate()

