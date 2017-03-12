import yaml,random
from metsub import *
from dbutils import *
def cleanup():
    for objname in sorted(globals()):
        try:
            obj = eval(objname)
            if isinstance(obj,SingleWordData):
                obj.destroy()
            elif isinstance(obj,MetaphorSubstitute):
                for sobjname in dir(obj):
                    sobj = eval(objname+'.'+sobjname)
                    if isinstance(sobj,SingleWordData):
                        sobj.destroy()
        except:
            None

def pairblock(adj,noun,dat,is_dry=False):
    return  {
        'pred' : adj,
        'noun' : noun,
        'topfour' : dat.get('neuman_top_four'),
        'mode' : dat.get('mode'),
        'gold' : dat['gold'],
        'coca_syns' : pairs[adj]['coca_syns'],
        'roget_syns' : pairs[adj]['roget_syns'],
        'dry_run' : is_dry,
        'semid' : dat['id'] 
    }

if __name__ == '__main__':
    processed_pairs = sum([[pairblock(adj,noun,ndata) for noun,ndata in pairs[adj]['with'].items()] for adj in pairs.keys()],[])
    processed_pairs.sort(key=lambda x: x['noun'])
    processed_pairs.sort(key=lambda x: x['pred'])
    print("adjective - object pairs:")
    for i,p in enumerate(processed_pairs,1):
        print(str(i)+".",p['pred'],p['noun'])
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
        run.append(processed_pairs[pair])
    
    if mode == "2":
        random.shuffle(processed_pairs)
        num = int(input("how many: "))
        run = processed_pairs[:num]
    
    if mode == "3":
        run = processed_pairs
    
    if mode == "4":
        adj = ""
        while adj not in pairs:
            adj =  input("pick an adjective: ")
        run = [b for b in processed_pairs if b['pred'] == adj]
    
    rundata = list()
    note = input("add a note about this run:")
    conf.update({
        'methods' : {
            'candidates' : 'ngramcands(50)',
            'rating' : 'by_coca_synonyms_of_pred(15)'
        }
    })
    for p in run:
        conf.update(p)
        ms = AdjSubstitute(conf)
        #cands = ms.get_candidates()
        subs = ms.find_substitutes()
        d = {
            "pred": p['pred'],
            "noun" : p['noun'],
            "substitutes" : subs,
            "neuman_score" : ms.neuman_eval(),
            "avprec" : ms.AP(),
            "mode" : ms.mode,
            "strictprec" : ms.strictP(),
            "spearmanr" : ms.spear(),
            "overlap" : ms.overlap(),
            "top_in_gold" : ms.lenient_acc(),
            "none_in_gold" : ms.complete_miss(),
            "cands_size" : len(subs),
            "semid" : ms.semid
        }
        rundata.append(d)
        print(ms,"\n====================\n")
        #ovl = overlap(cands,set(p['gold']))
        #if ovl == 0:
        #    print(p['pred'],p['noun'],printlist(cands,20),printlist(p['gold']))
        #rundata.append({'oc': ovl ,'size' : len(cands)})
    #data = pandas.DataFrame(rundata)
    rundata = RunData(pandas.DataFrame(rundata),note,ms.classname)
    print("wrapping up and saving stuff")
    if note != "discard":
        rundata.save()
    cleanup()
    print("\n\t\tHura!\n")


