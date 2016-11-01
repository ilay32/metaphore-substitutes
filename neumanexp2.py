import os,pickle,yaml,re,copy
from ngraph import NeumanGraph
from metsub import *
pairs = yaml.load(open('adj_pairs.yml'))
adjsyns = dict()
centroids = dict()
nounadjs = ObjectAdjs(ngrams) 

class NeumanExp2:
    def __init__(self,adj,noun):
        self.adj = adj
        self.noun = noun
        #self.options = pairs[adj]['with'][noun]['neuman_top_four']
        #self.options = pairs[adj]['coca_syns'] + [pairs[adj]['with'][noun]['correct']]
        self.options = self.get_options()
        self.results = {"in":"","out" : "","all" : ""}
        self.correct = pairs[adj]['with'][noun]['correct']
        iadjs = [a for a in nounadjs.get(noun) if (abst.has(a) and vecs.has(a))]
        self.instance_adjs = sorted(iadjs,key=lambda x: abst.get(x),reverse=True)
    
    def get_graph_data(self,kind,crit):
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,self.adj+"-"+kind+".pkl"),'rb'))
        proto_nouns = [n[0] for n in data[crit].most_common() if vecs.has(n[0])][:NeumanGraph.most_connected]
        return MetaphorSubstitute.cluster_centroid(proto_nouns)
    
    def get_options(self):
        return pairs[self.adj]['with'][self.noun]['neuman_top_four']
        raw = sorted([a for a in nounadjs.get(self.noun) if abst.has(a)][:100],key=lambda x: abst.get(x),reverse=True)[:10]
        raw.append(pairs[self.adj]['with'][self.noun]['correct'])
        return raw

    def get_synonyms(self):
        return sorted([a for a in pairs[self.adj]['coca_syns'] if vecs.has(a)],key=lambda x: abst.get(x),reverse=True)[:NeumanGraph.most_connected]
        #return [a for a in pairs[self.adj]['coca_syns'] if vecs.has(a) and abst.get(a) > abst.get(self.adj)][:NeumanGraph.most_connected]

        if self.adj in adjsyns :
            s = copy.copy(adjsyns[self.adj])
            print("number of synonyms:",len(s))
            return s
        simcut =  0.1 # Theta_cut
        syns = [a for a in pairs[self.adj]['coca_syns'] if vecs.has(a)][:NeumanGraph.most_connected]
        ret = list()
        for s in syns:
            sim_abst = Vecs.distance(self.abstract_vector,vecs.get(s))
            sim_conc = Vecs.distance(self.concrete_vector,vecs.get(s))
            if sim_abst > simcut and sim_conc < simcut:
                ret.append(s)
            else:
                print(s,"ruled out by protolist")
        print("in total ruled out for", self.adj+":",len(syns) - len(ret))
        ret.append(self.adj)
        print(ret)
        adjsyns[self.adj] = ret
        return self.get_synonyms()
    
    def rate_substitutes(self):
        c = self.get_synonyms()
        #c.append(self.noun)
        #c += self.instance_adjs[:5]
        mingled = MetaphorSubstitute.cluster_centroid(c)
        ret = [(sub,Vecs.distance(vecs.get(sub),mingled)) for sub in self.options]
        return sorted(ret,key=lambda x: x[1],reverse=True)
    
    def eval_results(self,ccriterion):
        #self.concrete_vector = self.get_graph_data('concrete',ccriterion)
        #self.abstract_vector = self.get_graph_data('abstract',ccriterion)
        res = self.rate_substitutes()
        self.results[ccriterion] = res[0][0]
        return int(res[0][0] == self.correct)
    
    
if __name__ == '__main__':
    rundata = list()
    for adj in pairs:
        for noun in pairs[adj]['with']:
            print(adj,noun)
            ne2 = NeumanExp2(adj,noun)
            correct = ne2.correct            
            rundata.append({    
                "adj" : adj,
                "noun" : noun,
                #"in": ne2.eval_results('in'),
                #"out": ne2.eval_results('out'),
                "all": ne2.eval_results('all'),
                "expected" :correct,
                "got" : re.sub('[\n\t]',' ', printdict(ne2.results,True)),
                #"no_vec" : int(not vecs.has(correct))
            }) 
    data = pandas.DataFrame(rundata)
    vecs.destroy()
    abst.destroy()
    nounadjs.destroy()

