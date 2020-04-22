# import thinqpbo as tq
import pdb
import json
import argparse
import rouge
from rouge import rouge_n_sentence_level
import math
import thinqpbo as tq
# node_t = 0.65
# edge_t = 0.85
node_t = 0.65
edge_t = 0.85
# node_t = 0.1
# edge_t = 0.003
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_fp')
    parser.add_argument('--out_fp')

    return parser

def qpbo(out_fp, sent_list, graphD, node_edgeD):
    # Create graph object.
    # Number of nodes to add.
    f = open(out_fp, "w")
    cnt = 0
    for i in range(len(sent_list)):
        graph = tq.QPBODouble()
        nodes_to_add = len(graphD[sent_list[i]])
        num = nodes_to_add
        first_node_id = graph.add_node(nodes_to_add)
        visit = dict()
        for j in range(num):
            node_score = node_edgeD[sent_list[i]][j]
            # print(node_score)
            if node_score < node_t:
                visit[j] = False
                continue
            visit[j] = True
            # print(graphD[sent_list[i]][j], node_score)
            graph.add_unary_term(j, 0, -node_edgeD[sent_list[i]][j])
        l1 = list(visit.values())
        if True in l1:
            cnt += 1
        for j in range(num):
            for k in range(j+1, num):
                edge_score = node_edgeD[sent_list[i]][(j,k)]
                # print(edge_score)
                if not((visit[j] == True) and (visit[k] == True) and (edge_score > edge_t)):
                    continue
                # print(j, k, edge_score)
                graph.add_pairwise_term(j, k, 0, 0, 0, edge_score)      
        graph.solve()
        #if i==0:
        #   pdb.set_trace()
        graph.compute_weak_persistencies()
        twice_energy = graph.compute_twice_energy()
        for n in range(nodes_to_add):
            segment = graph.get_label(n)
            # print(i,segment)
            if segment == 1:
                f.write(sent_list[i] + '\t' + graphD[sent_list[i]][n] + '\t' + str(math.log(node_edgeD[sent_list[i]][n])) + '\n')
    # Add two nodes.
    # Add edges.
    print(cnt)
    return  

def get_data(inp_fp):
    inp_f = open(inp_fp,'r')
    extD = dict()
    graphD = dict()
    node_edgeD = dict()
    for line in inp_f:
        line = line.strip('\n')
        sentence, extraction, confidence = line.split('\t')
        if sentence not in extD:
            extD[sentence] = list()

        already_added = False
        for added_extraction, _ in extD[sentence]:
            if extraction == added_extraction:
                already_added = True
        if already_added:
            continue

        extD[sentence].append([extraction, confidence])
    for key in extD.keys():
        graphD[key] = dict()
        cnt = 0
        for item in extD[key]:
            graphD[key][cnt]=item[0]
            cnt += 1
    sent_list = []
    # sent_dict = json.load(open('extractions.txt', 'r'))
    sent_dict = extD
    for key in sent_dict:
        sent_list.append(key)
        node_edgeD[key] = dict()
        num = len(sent_dict[key])
        key_sum = 0
        for i in range(num):
            # for j in range(num):
            #     if sent_dict[key][i] == extD[key][j][0]:
                    # key_sum += math.exp(float(extD[key][j][1]))
            # node_edgeD[key][i] = math.exp(float(extD[key][i][1]))
            key_sum += math.exp(float(extD[key][i][1]))

        for i in range(num):
            # for j in range(num):
            #     if sent_dict[key][i] == extD[key][j][0]:
            #         node_edgeD[key][i] = str((math.exp(float(extD[key][j][1])))/key_sum)
            
            # node_edgeD[key][i] = (math.exp(float(extD[key][i][1])))/key_sum
            node_edgeD[key][i] = (math.exp(float(extD[key][i][1])))
                    
        edge_sum = 0
        for i in range(0,num):
            for j in range(i+1, num):
                # sent1 = sent_dict[key][i].split()
                # sent2 = sent_dict[key][j].split()
                sent1 = ''.join(sent_dict[key][i])
                sent2 = ''.join(sent_dict[key][j])
                
                recall, precision, rouge = rouge_n_sentence_level(sent1, sent2, 2)
                #node_edgeD[key][(i,j)] = rouge
                edge_sum += rouge
        for i in range(0,num):
            for j in range(i+1, num):
                # sent1 = sent_dict[key][i].split()
                # sent2 = sent_dict[key][j].split()
                sent1 = ''.join(sent_dict[key][i])
                sent2 = ''.join(sent_dict[key][j])
                recall, precision, rouge = rouge_n_sentence_level(sent1, sent2, 2)

                # node_edgeD[key][(i,j)] = rouge/edge_sum
                node_edgeD[key][(i,j)] = rouge
    return sent_list, extD, graphD, node_edgeD
    

def main():
    parser = parse_args()
    args = parser.parse_args()
    sent_list, extD, graphD, node_edgeD = get_data(args.inp_fp)
    # print(node_edgeD)
    # with open('data2.txt', 'w') as outfile:
    #    json.dump(node_edgeD, outfile)
    qpbo(args.out_fp, sent_list, graphD, node_edgeD)
    #pdb.set_trace()                            

if __name__ == '__main__':
    main()  

