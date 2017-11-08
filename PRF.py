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

ROCCHIO_ALPHA = 0.7
ROOCHIO_BETA = 0.3
TMM_ALPHA = 0.3
TMM_BETA = 0.3
KL_A = 0.5
KL_B = 0.0
KL_D = 0.7

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
            temp = copy.deepcopy(DOCUMENT[DOC_NAME.index(d_name)])
            doc_dict.append(temp)
            for temp_word in temp:
                if temp_word not in total_voc:
                    total_voc[temp_word] = temp[temp_word]
                else:
                    total_voc[temp_word] += temp[temp_word]

        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    tmm_list = copy.deepcopy(relevant_doc_word)

    for iteration in range(1, 21):
        print(iteration)
        for q in range(len(relevant_doc_word)):  # query 800
            tmm = copy.deepcopy(tmm_list[q])
            if iteration == 1:
                # for tmm_voc in tmm:
                #     tmm[tmm_voc] = random.random()
                tmm_total = sum(tmm.values())
                for tmm_voc in tmm:
                    tmm[tmm_voc] /= tmm_total
            tmmwd = copy.deepcopy(relevant_doc[q])  # 只是剛好relevant_doc[q] = q all file dictionary
            tdwd = copy.deepcopy(relevant_doc[q])   # 同上

            # E_Step
            for doc_id, doc in enumerate(relevant_doc[q]):
                doc_total = sum(doc.values())
                for word_id, word in enumerate(doc):
                    pbg = 0
                    pwd = doc[word] / doc_total
                    if int(word) < len(BG):
                        pbg = math.exp(BG[int(word)])
                    twd_deno = ((tmm[word] * (1 - TMM_ALPHA - TMM_BETA)) + (pwd * TMM_ALPHA) + (pbg * TMM_BETA))
                    tmmwd[doc_id][word] = (tmm[word] * (1 - TMM_ALPHA - TMM_BETA)) / twd_deno
                    tdwd[doc_id][word] = (pwd * TMM_ALPHA) / twd_deno

            # M_Step
            tmm_molecular_list = []
            for word_id, word in enumerate(tmm):
                molecular = sum(tmmwd[doc_id][word] * doc[word] for doc_id, doc in enumerate(relevant_doc[q]) if word in doc)
                tmm_molecular_list.append(molecular)
            denominator = sum(tmm_molecular_list)

            for word_id, word in enumerate(tmm):
                tmm[word] = tmm_molecular_list[word_id] / denominator
            tmm_list[q] = tmm

            for doc_id, doc in enumerate(relevant_doc[q]):
                td_molecular_list = []
                for word in doc:
                    molecular = tdwd[doc_id][word] * doc[word]
                    td_molecular_list.append(molecular)
                denominator = sum(td_molecular_list)
                for word_id, word in enumerate(doc):
                    relevant_doc[q][doc_id][word] = td_molecular_list[word_id] / denominator

            # print(sum(tmm.values()))
            # print(sum(relevant_doc[q][0].values()))

        l = 1
        for doc_id, doc in enumerate(relevant_doc[0]):
            word_total = sum(doc.values())
            for word_id, word in enumerate(doc):
                l += math.log10(pow(((1 - TMM_ALPHA - TMM_BETA) * tmm_list[0][word]) + (TMM_ALPHA * doc[word]) + (TMM_BETA * math.exp(BG[int(word)])), doc[word]))
        print(math.exp(l))
    return tmm_list


def KL(tmm_query):
    global QUERY, DOCUMENT, RANKING, DOC_NAME,BG
    global KL_A, KL_B, KL_D
    for q_id, q in enumerate(QUERY):
        if q_id % 100 == 0:
            print(q_id)

        kl_list = []
        tmq = copy.deepcopy(tmm_query[q_id])

        q_total = sum(q.values())
        new_q = copy.deepcopy(q)
        new_q.update(tmq)

        for q_word in new_q:
            base_q, tmm, pbg = 0, 0, 0
            new_q[q_word] = 0

            if q_word in tmq:
                tmm = tmq[q_word]
            if q_word in q:
                base_q = q[q_word]
            if int(q_word) < len(BG):
                pbg = math.exp(BG[int(q_word)])

            new_q[q_word] = (KL_A * base_q / q_total) + (KL_B * tmm) + ((1 - KL_A - KL_B) * pbg)
        #KL
        for doc in DOCUMENT:
            doc_total = sum(doc.values())
            # kl_score = -sum(new_q[q_word] * math.log((KL_D * doc[q_word] / doc_total) + ((1 - KL_D) * math.exp(BG[int(q_word)])), math.e)
            #                 for q_word in new_q if q_word in doc)
            kl_score = 0
            for q_word in new_q:
                if q_word in doc:
                    new_doc = (KL_D * doc[q_word] / doc_total) + ((1 - KL_D) * math.exp(BG[int(q_word)]))
                    kl_score -= (new_q[q_word]) * math.log10(new_doc)
            kl_list.append(kl_score)

        #sorting
        sort = sorted(kl_list, reverse=True)
        q_rank = []
        for sort_num in sort:
            q_rank.append(DOC_NAME[kl_list.index(sort_num)])

        RANKING.append(q_rank)


def QL(tmm_query):
    global QUERY, DOCUMENT, RANKING, DOC_NAME, BG
    global KL_A, KL_B, KL_D

    for q_id, q in enumerate(QUERY):
        if q_id % 100 == 0:
            print(q_id)

        tmq = copy.deepcopy(tmm_query[q_id])
        q_total = sum(q.values())
        new_q = copy.deepcopy(q)
        new_q.update(tmq)

        for q_word in new_q:
            base_q, tmm, pbg = 0, 0, 0
            new_q[q_word] = 0

            if q_word in tmq:
                tmm = tmq[q_word]
            if q_word in q:
                base_q = q[q_word]
            if int(q_word) < len(BG):
                pbg = math.exp(BG[int(q_word)])
            new_q[q_word] = (KL_A * base_q / q_total) + (KL_B * tmm) + ((1 - KL_A - KL_B) * pbg)

        score_list = []
        for doc in DOCUMENT:
            doc_total = sum(doc.values())
            score = 1
            for q_word in new_q:
                if q_word in doc:
                    score *= ((0.7 * doc[q_word] / doc_total) + (0.3 * math.exp(BG[int(q_word)])))
            score_list.append(score)

        sort = sorted(score_list, reverse=True)
        q_rank = []
        for sort_num in sort:
            q_rank.append(DOC_NAME[score_list.index(sort_num)])

        RANKING.append(q_rank)


def relevant_doc(first):
    global QUERY
    relevant_doc = []
    relevant_doc_word = []
    new_q = []

    for d_list in first:
        doc_dict = []
        total_voc = {}
        for d_name in d_list:
            temp = copy.deepcopy(DOCUMENT[DOC_NAME.index(d_name)])
            doc_dict.append(temp)
            for temp_word in temp:
                if temp_word not in total_voc:
                    total_voc[temp_word] = temp[temp_word]
                else:
                    total_voc[temp_word] += temp[temp_word]

        relevant_doc.append(doc_dict)     # query k 的 relevant doc
        relevant_doc_word.append(total_voc)     # query k relevant doc all word

    # for q_id, q in enumerate(QUERY):
    #     pseudo = q.copy()
    #
    #     for doc in relevant_doc[q_id]:
    #         for relevant_word in doc:
    #             if relevant_word not in pseudo:
    #                 pseudo[relevant_word] = doc[relevant_word]
    #             else:
    #                 pseudo[relevant_word] += doc[relevant_word]
    #     new_q.append(pseudo)
    return relevant_doc_word


def writefile(name):
    global QUERY_NAME, DOC_NAME, RANKING
    with open(name + '.txt', 'w') as retrieval_file:
        retrieval_file.write("Query,RetrievedDocuments\n")
        for retrieval_id, retrieval_list in enumerate(RANKING):
            retrieval_file.write(QUERY_NAME[retrieval_id] + ',')
            for retrieval_name in retrieval_list[0:100]:
                retrieval_file.write(retrieval_name + ' ')
            if retrieval_id != len(QUERY_NAME) - 1:
                retrieval_file.write('\n')


readfile()
# VSM_re = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=1)
# VSM_re.calculate()
# VSM_re.writeAns('VSM5')
print('VSM down')
answer = ans_read('VSM\\VSM1.txt')
# tmm_query = TMM(answer)
# print('TMM down')
# KL(tmm_query)
# writefile('KL')
# print('KL down')

# QL(tmm_query)
# writefile('QL')
# print('QL down')


new_que = relevant_doc(answer)
VSM_ro = VSM(doc_name=DOC_NAME, query_name=QUERY_NAME, document=DOCUMENT, query=QUERY, rank_amount=100)
VSM_ro.rocchioauto(0.6, 0.4, new_que, 1)
VSM_ro.writeAns('roteee2')


print('Process down')
#
