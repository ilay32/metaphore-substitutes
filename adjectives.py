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
    print(noun)
    if 'gold_freqs' in dat:
        gldfrqs = dat['gold_freqs']
    else:
        gldfrqs = list(range(1,len(dat['gold'])+1))
        gldfrqs.reverse()
    return  {
        'pred' : adj,
        'noun' : noun,
        'topfour' : dat.get('neuman_top_four'),
        'neuman_correct' : dat.get('neuman_correct'),
        'gold' : dat['gold'],
        'gold_rates' : gldfrqs,    
        'coca_syns' : pairs[adj]['coca_syns'],
        'roget_syns' : pairs[adj]['roget_syns'],
        'dry_run' : is_dry,
        'semid' : dat.get('id')
    }

processed_pairs = sum([[pairblock(adj,noun,ndata) for noun,ndata in pairs[adj]['with'].items()] for adj in pairs.keys()],[])
processed_pairs.sort(key=lambda x: x['noun'])
processed_pairs.sort(key=lambda x: x['pred'])

def main(fetcher,rater,classname='AdjSubstitute'):
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
    conf = {
        'methods' : {
            'candidates' : fetcher,
            'rating' : rater #irrelevant for Irst2
        }
    }
    for p in run:
        conf.update(p)
        ms = eval(classname)(conf)
        rundata.append(ms.export_data())
        print("\n====================\n")
    rundata = RunData(pandas.DataFrame(rundata),note,ms.classname)
    print("wrapping up and saving stuff")
    if note != "discard":
        rundata.save()
    cleanup()
    print("\n\t\tHura!\n")
    return rundata


if __name__ == '__main__':
    rundata = main('neumans_four()','neuman_orig()')
