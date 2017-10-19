import yaml,random,itertools,copy
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
        'semid' : dat.get('id'),
        'orign_touchstone' : pairs[adj].get('orign_touchstone')
    }

processed_pairs = sum([[pairblock(adj,noun,ndata) for noun,ndata in pairs[adj]['with'].items()] for adj in pairs.keys()],[])
processed_pairs.sort(key=lambda x: x['noun'])
processed_pairs.sort(key=lambda x: x['pred'])

def main(fetcher,rater,classname='AdjSubstitute',system_name=None):
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
    if classname == 'SemEvalSystem': 
        if  system_name is None:
            print("SemEvalSystem requires a system name")
            quit()
        else:
            conf['system_name'] = system_name

    for p in run:
        conf.update(p)
        ms = eval(classname)(conf)
        rundata.append(ms.export_data())
        print("\n====================\n")
    rundata = RunData(pd.DataFrame(rundata),note,ms.classname)
    print("wrapping up and saving stuff")
    if note != "discard":
        rundata.save()
    #cleanup()
    print("\n\t\tHura!\n")
    return rundata

def mindat(d):
    return [d['gap'],d['overlap']]

if __name__ == '__main__':
    norig = main('all_dicts_and_coca(2,15)','neuman_orig()')
    imp = main('all_dicts_and_coca(2,15)','neuman_no_filtering()')
    #ut = main('all_dicts_and_coca(2,15)','utsumi1cat(10)')
    #irst = main('semgold()','neuman_orig()','Irst2')
    #conf = {
    #    'methods' : {
    #        'candidates' : 'semgold()',
    #    }
    #}

    #dat = pd.DataFrame(index=pd.MultiIndex.from_product([['SimpNeum','Irst2','Uts1Cat','Dagan'],[p['semid'] for p in processed_pairs]]),columns=['gap','coveragcoveragee'])
    #for p in processed_pairs:
    #    semid = p['semid']
    #    conf.update(p)
    #    conf['methods']['rating'] = 'neuman_no_filtering()'
    #    ms = AdjSubstitute(conf)
    #    dat.loc['SimpNeum',semid] = mindat(ms.export_data())
    #    irst = Irst2(conf)
    #    dat.loc['Irst2',semid] = mindat(irst.export_data())
    #    conf['methods']['rating'] = 'utsumi1cat(10)'
    #    uts = AdjSubstitute(conf)
    #    dat.loc['Uts1Cat',semid] = mindat(uts.export_data())
    #    conf['system_name'] = 'DF'
    #    dag = SemEvalSystem(conf)
    #    dat.loc['Dagan',semid] = mindat(dag.export_data())
    #cleanup()
