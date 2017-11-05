import os
import re
import random
import math
import copy
from Vector_Space_Model import VSM
import numpy as np

DOC_NAME = os.listdir("Document")  # Document file name
QUERY_NAME = os.listdir("Query")  # Query file name

QUERY = []
DOCUMENT = []
BG = []

TMM_ALPHA = 0.3
TMM_BETA = 0.3

RANKING = []

def readfile():
    global QUERY, DOCUMENT, BG
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

    # Load BG
    with open('BGLM.txt') as BG_file:
        for len_voc_f, val_voc in enumerate(BG_file):
            bg_split = re.split('   |\n', val_voc)
            BG.append(float(bg_split[1]))

    print('read file down')


def ans_read(ans):

    ans_list = []
    with open(ans) as ans_file:
        for line in ans_file:
            if line == 'Query,RetrievedDocuments\n':
                continue
            ans_name = re.split(',| ', line)
            ans_name.remove('\n')
            ans_name.pop(0)
            ans_list.append(ans_name)
    return ans_list


def TMM(first):
    global TMM_ALPHA, TMM_BETA
    global DOCUMENT, DOC_NAME, BG
    relevant_doc = []
    relevant_doc_word = []

    for d_list in first:
        doc_dict = []
        total_voc = {}
        for d_name in d_list:
            temp = DOCUMENT[DOC_NAME.index(d_name)]
            doc_dict.append(temp)
            total_voc.update(temp)
        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    tmm_list = copy.deepcopy(relevant_doc_word)

    for iteration in range(1, 21):
        print(iteration)
        for q in range(len(relevant_doc_word)):  # query 800
            tmm = copy.deepcopy(tmm_list[q])
            # if iteration == 1:
            #     for tmm_voc in tmm:
            #         tmm[tmm_voc] = random.random()
                # tmm_total = sum(tmm.values())
                # for tmm_voc in tmm:
                #     tmm[tmm_voc] /= tmm_total
            twd = copy.deepcopy(relevant_doc[q])

            # E_Step
            for doc_id, doc in enumerate(relevant_doc[q]):
                doc_total = sum(doc.values())
                for word_id, word in enumerate(doc):
                    pbg = 0
                    pwd = doc[word] / doc_total
                    if int(word) < len(BG):
                        pbg = math.exp(BG[int(word)])
                    twd[doc_id][word] = tmm[word] * TMM_ALPHA / ((tmm[word] * TMM_ALPHA) + (pwd * TMM_BETA) + (pbg * (1 - TMM_ALPHA - TMM_BETA)))
            # M_Step
            molecular_list = []
            for word_id, word in enumerate(tmm):
                molecular = sum(doc[word] * twd[doc_id][word] for doc_id, doc in enumerate(relevant_doc[q]) if word in doc)
                molecular_list.append(molecular)
            denominator = sum(molecular_list)

            for word_id, word in enumerate(tmm):
                tmm[word] = molecular_list[word_id] / denominator
            tmm_list[q] = tmm

    return tmm_list


def KL(tmm_query):
    global QUERY, DOCUMENT, RANKING, DOC_NAME
    for q_id, q in enumerate(QUERY):
        if q_id % 100 == 0:
            print(q_id)
        kl_list = []
        # qaddtq = tmm_query[q_id].copy()
        # qaddtq.update(q)
        # qaddtq_total = sum(qaddtq.values())
        q_total = sum(q.values())
        tq = tmm_query[q_id]
        tq_total = sum(tq.values())
        #KL
        for doc in DOCUMENT:
            doc_total = sum(doc.values())
            # kl_score = sum(-((qaddtq[q_word] / qaddtq_total) * math.log10(doc[q_word] / doc_total))
            #                for q_word in qaddtq if q_word in doc)
            kl_score = sum(-((q[q_word] / q_total) * math.log10(doc[q_word] / doc_total)) for q_word in q if q_word in doc)
            kl_score += sum(-((tq[q_word] / tq_total) * math.log10(doc[q_word] / doc_total)) for q_word in tq if q_word in doc)
            kl_list.append(kl_score)
        #sorting
        sort = sorted(kl_list, reverse=True)
        q_rank = []
        for sort_num in sort:
            q_rank.append(DOC_NAME[kl_list.index(sort_num)])

        RANKING.append(q_rank)



def writefile():
    global QUERY_NAME, DOC_NAME, RANKING
    with open('test3.txt', 'w') as retrieval_file:
        retrieval_file.write("Query,RetrievedDocuments\n")
        for retrieval_id, retrieval_list in enumerate(RANKING):
            retrieval_file.write(QUERY_NAME[retrieval_id] + ',')
            for retrieval_name in retrieval_list[0:100]:
                retrieval_file.write(retrieval_name + ' ')
            if retrieval_id != len(QUERY_NAME) - 1:
                retrieval_file.write('\n')


readfile()
# VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=5)
# VSM_re.calculate()
# VSM_re.writeAns('VSM5')
print('VSM down')
answer = ans_read('VSM5.txt')
first_rank = answer
tmm_query = TMM(first_rank)
print('TMM down')
KL(tmm_query)
print('KL down')
writefile()
print('Process down')
#
