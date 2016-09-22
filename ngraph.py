# implementaion of the graph described in Neuman et al 2011
from prygress import progress
from dbutils import *
import nltk,os,pickle
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
objects = AdjObjects(ngrams)
nouns = NounNoun(ngrams)
class NeumanGraph:
    #clust_size = 1000 # Theta_num using params.cluster_maxrows
    #adj_window = 1 # Theta_c params.search_aobjects_right will be used
    #mi = 3 # Theta_mi params.noun_min_MI will be used
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
        if not os.path.isdir(NeumanGraph.datadir):
            os.makedirs(NeumanGraph.datadir)
    
    @hereiam
    def get_master_list(self):
        adjnouns = objects.get(self.adj)
        adjnouns.sort(key = self.master_list_criterion())
        return adjnouns[:min(len(adjnouns),NeumanGraph.top_n)]


    def master_list_criterion(self):
        def kcmp(a):
            if abst.has(a):
                return self.cmpfactor * abst.get(a)
            return 0.5
        return kcmp
    
    @hereiam
    def construct_graph(self):
        m = self.get_master_list()
        print("master list:",printlist(m,20,True))
        for mn in m:
            print("getting",mn+"'s neighbours")
            neighbors = sorted(nouns.get(mn),key=self.master_list_criterion())
            print("found:",printlist(neighbors,10,True))
            for n in neighbors[:min(len(neighbors),NeumanGraph.top_n)]:
                if n in m and n != mn:
                    self.vertices.add(n)
                    self.vertices.add(mn)
                    self.edges.add((mn,n))
                    print("added",mn,n)
    
    def compile_graph_data(self):
        d = nltk.ConditionalFreqDist()
        for e in self.edges:
           d['all'][e[0]] += 1
           d['all'][e[1]] += 1
           d['in'][e[1]] += 1
           d['out'][e[0]] += 1
        with open(os.path.join(NeumanGraph.datadir,self.adj+"-"+self.kind+".pkl"),'wb') as p:
            pickle.dump(d,p)

if __name__ == '__main__':
    for adj in ['dark','hard','sweet','warm','deep']:
        for kind in ['abstract','concrete']:
            g = NeumanGraph(adj,kind)
            g.construct_graph()
            g.compile_graph_data()
    nouns.destroy()
    abst.destroy()
    objects.destroy()
    print("done")
