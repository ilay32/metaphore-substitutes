import yaml,random
from metsub import *
from dbutils import *

def pairblock(adj,noun,dat):
    return {
        'pred' : adj,
        'noun' : noun,
        'topfour' : dat['neuman_top_four'],
        'correct' : dat['correct']
    }

if __name__ ==  '__main__':
    pairsraw = yaml.load(open('adj_pairs.yml'))
    pairs = sum([[pairblock(adj,noun,ndata) for noun,ndata in pairsraw[adj]['with'].items()] for adj in pairsraw.keys()],[])
    pairs.sort(key=lambda x: x['noun'])
    pairs.sort(key=lambda x: x['pred'])
    print("adjective - object pairs:")
    for i,p in enumerate(pairs):
        print(str(i+1)+".",p['pred'],",",p['noun'])
    print("modes:")
    print("1. number -- pick a pair of the above")
    print("2. rnd n randomly pick n pairs")
    print("3. all")
    print("4. adj -- pick an adjective")
    
    run = list()
    mode = ""
    while mode not in ["1","2","3","4"]:
        mode = input("choose mode: ")
    if mode == "1":
        pair = int(input("pick a pair by it's number on the list: ")) - 1
        run.append(pairs[pair])
    
    if mode == "2":
        random.shuffle(pairs)
        num = int(input("how many: "))
        run = pairs[:num]
    
    if mode == "3":
        run = pairs
    
    if mode == "4":
        adj = ""
        while adj not in pairsraw:
            adj =  input("pick an adjective: ")
        run = sorted([pairblock(adj,noun,ndata) for noun,ndata in pairsraw[adj].items()],key=lambda x : x['noun'])
    
    rundata = list()
    note = input("add a note about this run:")
    for p in run:
        conf.update(p)
        ms = AdjWithGraphProt(conf)
        if ms.go:
            subs = ms.find_substitutes()
            d = {
                "pred": p['pred'],
                "noun" : p['noun'],
                "substitutes" : subs, 
                "no_vector" : ms.no_vector, 
                "no_abst" : ms.no_abst,
                "too_far" : ms.too_far,
                "neuman_score" : ms.neuman_eval()
            }
            rundata.append(d)
        print(ms," done","\n====================\n")
    rundata = RunData(pandas.DataFrame(rundata),note,'adjecitves')
    print("wrapping up and saving stuff")
    if note != "discard":
        rundata.save()
    vecs.destroy()
    ngrams.destroy()
    abst.destroy()
    lex.destroy()
    AdjSubstitute.object_clusters.destroy()
    print("\n\t\tHura!\n")


