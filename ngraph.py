# implementaion of the graph described in Neuman et al 2011
from prygress import progress
from dbutils import *
import nltk,os,pickle
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
objects = AdjObjects(ngrams)
class NeumanGraph:
    clust_size = 1000 # Theta_num using params.cluster_maxrows
    adj_window = 1 # Theta_c params.search_aobjects_right will be used
    mi = 3 # Theta_mi params.noun_min_MI will be used
    top_n =  50 # Kappa
    nn_window = 3 # Theta_c2
    most_connected = 10 # Theta_top
    datadir = "graph-data"
    
    def __init__(self,adj,kind):
        self.adj = adj
        self.kind = kind
        self.vertices = set()
        self.edges = set()
        self.cmpfactor = -1 if kind == 'abstract' else 1
        self.savefile = os.path.join(NeumanGraph.datadir,self.adj+"-"+self.kind+".pkl")
        self.nouns = NounNoun(DB('NgramsCOCAAS'))
        if not os.path.isdir(NeumanGraph.datadir):
            os.makedirs(NeumanGraph.datadir)
    
            
    @hereiam
    def get_master_list(self):
        adjnouns = objects.get(self.adj)
        if adjnouns:
            nouns = [n[0] for n in adjnouns.most_common()]
            nouns.sort(key = self.master_list_criterion())
            return nouns[:min(len(nouns),NeumanGraph.top_n)]
        return False

    def master_list_criterion(self):
        def kcmp(a):
            if abst.has(a):
                return self.cmpfactor * abst.get(a)
            return 0.5
        return kcmp
    
    def construct_graph(self):
        print("constructing",self.kind,"graph for",self.adj)
        print("=========================================")
        m = self.get_master_list()
        if not m:
            print("can't make master list for",self.adj, ". aborting.")
            self.nouns.destroy()
            return False
        print("master list:",printlist(m,20,True))
        for mn in m:
            print("getting",mn+"'s neighbours")
            try:
                neighbors = sorted([n[0] for n in self.nouns.get(mn).most_common],key=self.master_list_criterion())
                print("found:",printlist(neighbors,10,True))
                for n in neighbors[:min(len(neighbors),NeumanGraph.top_n)]:
                    if n in m and n != mn:
                        self.vertices.add(n)
                        self.vertices.add(mn)
                        self.edges.add((mn,n))
                        print("added",mn,n)
            except:
                print("can't find neighbors for",mn,"nothing added")
        print(self.adj,self.kind,"done.\n")
        return True

    def compile_graph_data(self):
        d = nltk.ConditionalFreqDist()
        for e in self.edges:
           d['all'][e[0]] += 1
           d['all'][e[1]] += 1
           d['in'][e[1]] += 1
           d['out'][e[0]] += 1
        with open(self.savefile,'wb') as p:
            pickle.dump(d,p)
        self.nouns.destroy()

if __name__ == '__main__':
    pairs = yaml.load(open('adj_pairs.yml'))
    for adj in pairs:
        print(adj)
        for cond in ['abstract','concrete']:
            tg = NeumanGraph(adj,cond)
            if tg.construct_graph() :
                tg.compile_graph_data()
        
        for d in pairs[adj]['with'].values():
            for cand in d['neuman_top_four']:
                print("\t",cand)
                g = NeumanGraph(cand,'abstract')
                if not os.path.isfile(g.savefile):
                    if g.construct_graph() :
                        g.compile_graph_data()
    abst.destroy()
    objects.destroy()
    print("done")
