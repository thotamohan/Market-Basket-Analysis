import json
import sys
import pyspark
import time
from operator import add
from pyspark import SparkContext
import itertools
from itertools import chain
from collections import defaultdict

def read_csv(x,k):
    x=x.split(',')
    if k==1:
        return (x[0],x[1])
    else:
        return (x[1],x[0])
    
def combinations(x,size):
    return itertools.combinations(x,size)

def more_combinations(x,size):
    all_items=list(set(itertools.chain.from_iterable(x)))
    comb=combinations(all_items,size)
    return comb

def check(baskets,candidate_set,reduced_threshold,size_count,freq_items):
    size_count={}
    freq_items=[]
    for value in candidate_set:
        set_value=set(value)
        for basket in baskets:
            if set_value.issubset(basket):
                if value in size_count:
                    size_count[value]+=1
                    if(size_count[value]>=reduced_threshold):
                        freq_items.append(tuple(sorted(value)))
                else:
                    size_count[value]=1
    return freq_items

def apriori_algorithmz1(support, baskets, totalitems,size):
    length=float(len(baskets))
    total=float(totalitems)
    modified_threshold = support * (length / total)
    candidate_itemset = []
    while True:
        if size == 1:
            candidate_items = []
            candidatecount = {}
            
            for basket in baskets:
                for item in basket:
                    if item in candidatecount:
                        candidatecount[item] += 1
                        if (candidatecount[item] >= modified_threshold):
                            candidate_items.append(item)
                    else:
                        candidatecount[item] = 1
                        
            if candidate_items:
                candidate_items=set(candidate_items)
                candidate_set=candidate_items
                for items in candidate_items:
                    candidate_itemset.append(tuple([items]))
        else:
            if size==2:
                candidate_set=combinations(candidate_set,size)
            else:
                candidate_set=more_combinations(candidate_set,size)
            
            comb_items = {}
            candidate_items=[]
            candidate_items=check(baskets,candidate_set,modified_threshold,comb_items,candidate_items)
            
            if candidate_items:
                candidate_items=set(candidate_items)
                candidate_set = candidate_items
                candidate_itemset.extend(candidate_items)
        size=size+1
        if not candidate_items:
            return candidate_itemset

def algorithm(baskets,candidate_set):
    frequent_items={}
    for item in candidate_set:
        for basket in baskets:
            if set(item).issubset(basket):
                if item in frequent_items:
                    frequent_items[item]+=1
                else:
                    frequent_items[item]=1
    return frequent_items.items()

if __name__ == "__main__":
    if len(sys.argv)!=5:
        print("This function needs 4 input arguments <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)
    
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path=sys.argv[3]
    output_file_path=sys.argv[4]
    
    

    sc = SparkContext("local[*]")
    start_time=time.time()
    items=sc.textFile(input_file_path).filter(lambda x:x!=('user_id,business_id')).map(lambda x:read_csv(x,case_number)).groupByKey().mapValues(set).map(lambda x:x[1])
    size=items.count()
    candidate_set=items.mapPartitions(lambda x:apriori_algorithmz1(support,list(x),size,1)).distinct().sortBy(lambda x:(len(x),x)).collect()
    frequent_items=items.mapPartitions(lambda x:algorithm(list(x),candidate_set)).reduceByKey(lambda value,n:value+n).filter(lambda x:x[1]>=support).map(lambda x:x[0]).sortBy(lambda x:(len(x),x)).collect()
    
    frequentItemSets= defaultdict(list)
    candidateItemSets=defaultdict(list)
    for x in frequent_items:
        frequentItemSets[len(x)].append(x)
    for x in candidate_set:
        candidateItemSets[len(x)].append(x)
    with open(output_file_path, 'w',encoding='utf-8') as outfile:
        outfile.write("Candidates:")
        outfile.write("\n")
        formatted=""
        for values in candidateItemSets.keys():
            if values==1:
                list_items=candidateItemSets[values]
                for items in list_items:
                    formatted+="('" + str(items[0]) + "'),"
            else:
                list_items=candidateItemSets[values]
                for items in list_items:
                    formatted+= str(items) + ","
            outfile.write(formatted[:-1])
            formatted=""
            outfile.write("\n")
            outfile.write("\n")
        #outfile.write("\n")
        outfile.write("Frequent Itemsets:")
        outfile.write("\n")
        formatted=""
        for values in frequentItemSets.keys():
            if values==1:
                list_items=frequentItemSets[values]
                for items in list_items:
                    formatted+="('" + str(items[0]) + "'),"
            else:
                list_items=frequentItemSets[values]
                for items in list_items:
                    formatted+= str(items) + ","
            outfile.write(formatted[:-1])
            formatted=""
            outfile.write("\n")
            outfile.write("\n")
    print('Duration:',time.time()-start_time)
    