import json
import sys
import pyspark
import time
from operator import add
from pyspark import SparkContext
import itertools
from itertools import chain
from collections import defaultdict
from pyspark import StorageLevel

def read_csv(x,k):
    x=x.split(',')
    if k==1:
        return (x[0],x[1])
    else:
        return (x[1],x[0])
    
def combinations(x,size):
    return itertools.combinations(x,size)

 
def more_size_candidate(k,Lists,mergings):     
    for i, j in enumerate(Lists[:-1]):
        for k1 in Lists[i+1:]:
            if j[:-1] == k1[:-1]:
                set_f1=set(j)
                set_f2=set(k1)
                L=list(set_f1 | set_f2)
                comb = tuple(sorted(L))
                values=[]
                for items in combinations(comb,k-1):
                    values.append(items)
                set_values=set(values)
                set_freq=set(Lists)
                if set_values.issubset(set_freq):
                    mergings.append(comb)  
            if j[:-1] != k1[:-1]:
                break             
    return mergings

def apriori(baskets, distinctItems, support, total):
    candidate_itemset = []
    k=0
    length = len(baskets)
    modified_threshold = int(support * (length / total))
    combin=distinctItems
    candidate_sets = {}
    
    while combin:
        dict_count={}
        k+=1
        tempcount = {}
        for basket in baskets:

            if k==1: 
                Single_combinations=combinations(basket,1)
                for value in Single_combinations:
                    if value in dict_count:
                        dict_count[value]=dict_count[value]+1
                    else:
                        dict_count[value]=1
                        
            elif k == 2: 
                thinbasket=sorted(list(set(basket).intersection(single_item))) 
                Double_combinations=combinations(thinbasket,2)
                for value in Double_combinations:
                    if value in dict_count:
                        dict_count[value]=dict_count[value]+1
                    else:
                        dict_count[value]=1
            
            else: 
                if len(basket)>=k:
                    for candy in combin:
                        set_candy=set(candy)
                        set_basket=set(basket)
                        if set_candy.issubset(set_basket):
                            if candy in dict_count:
                                dict_count[candy]=dict_count[candy]+1
                            else:
                                dict_count[candy]=1
                                               
                
        candidate_sets[k]=[]
        for keys,values in dict_count.items():
            if values>=modified_threshold:
                candidate_sets[k].append(keys)
        candidate_sets[k]=sorted(candidate_sets[k])
                
        if k > 1 :
            mergings=[]
            combin = more_size_candidate(k+1,candidate_sets[k],mergings)
        if k == 1:
            single_item=[]
            for item in candidate_sets[k]:
                single_item.append(item[0])
            combin=[]
            for pair in combinations(single_item, 2):
                combin.append(pair)
                    
        
    for values in candidate_sets.values():
        candidate_itemset.extend(values)
        
    yield candidate_itemset
    


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
    
    threshold_filter= int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path=sys.argv[3]
    output_file_path=sys.argv[4]
    
    sc = SparkContext("local[*]")
    start_time=time.time()
    File = sc.textFile(input_file_path)
    H = File.first()
    things = File.filter(lambda line: line!= H).map(lambda x:read_csv(x,1)).map(lambda t:tuple(t))\
    .groupByKey().mapValues(set).mapValues(sorted).mapValues(tuple).filter(lambda t:len(t[1])>threshold_filter).map(lambda t: t[1])

    
    
    t_set=set()
    for items in things.collect():
        for values in items:
            t_set.add(values)
    t_set=list(t_set)
    t_set=sorted(t_set)
    unique_items=[]
    for items in t_set:
        unique_items.append(tuple([items]))
    
    

    
    size = things.count()

    
    
    candidate_set=things.mapPartitions(lambda x:apriori(list(x), unique_items, support, size)).flatMap(lambda x:x).distinct().sortBy(lambda x:(len(x),x)).collect()
    frequent_items=things.mapPartitions(lambda x:algorithm(list(x),candidate_set)).reduceByKey(lambda value,n:value+n).filter(lambda x:x[1]>=support).map(lambda x:x[0]).sortBy(lambda x:(len(x),x)).collect()
   
    
    

    
    frequentItemSets= defaultdict(list)
    candidateItemSets=defaultdict(list)
    for x in frequent_items:
        frequentItemSets[len(x)].append(x)
    for x in candidate_set:
        candidateItemSets[len(x)].append(x)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
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