# main frame of the program #
import math
import numpy as np
from dbutils import *
vecs = Vecs(DB('LSA-ENG'))
Ngrams = DB('NgramsCOCAAS')
abst = Abst(Ngrams)
lex = Lex(DB('NgramsCOCAAS'))
class MetaphorSubstitute:
    def __init__ (self,options):
        for key in options.keys():
            self.__dict__[key] = options[key]
        self.instance_centroid = self.get_instance_centroid()
        self.substitutes = list()
    
    def __str__(self):
        return "verb:"+self.verb+", object:"+self.obj;
    
    # get n nouns, up to k words to the right of the verb,
    # in this version,simple filtering by the instance abstractness
    # threshold
    def object_cluster(self,verb):
        clust = list()
        q =  Ngrams.query("GetNgrams '{0}',{1},{2},{3},{4},{5}".format(
            verb,
            2, # this is the POS type of verb
            1, # POS of noun
            self.search_left, # how many words to look behind
            self.search_right, # how many words to look ahead
            self.min_MI # minimal mutual information required for results
        ))
        if not q:
            print("error fetching object cluster")
            return
        added = 0
        rows = q.fetchall()
        while added < min(self.object_cluster_size,len(rows) - 1):
            lem = rows[added][10] # take the lemma in this case ?
            clust.append(lem)
            added += 1
        return clust

    def cluster_centroid(self,cluster):
        added = 0
        l = len(vecs.get('queen')) # len of the vector is 300
        acc = np.zeros(l)
        # just sum the cluster words' vectors to get a semantic
        for word in cluster:
            if vecs.has(word):
                if math.isnan(vecs.get(word)[0]): 
                    print("nan in vector", word)
                acc = vecs.addition(acc,vecs.get(word))
                added += 1
        return acc/added
    
    def get_instance_centroid(self):
        raw_cluster = self.object_cluster(self.verb)
        return vecs.addition(self.cluster_centroid(raw_cluster),(vecs.get(self.obj)*self.instance_object_weight))/2
    
    # this is probably all wrong    
    def verb_synonyms(self):
        identifier = self.obj+"_"+self.verb+"\t"
        with open(self.synonyms_file) as s:
            for line in s:
                if identifier in line:
                    synraw = line.partition(identifier)[2].split(',')
                    print(synraw[:15])
                    synfiltered = [w for w in synraw if lex.has(w) and 2 in lex.get(w) and abst.has(w) and abst.get(w) > self.abstract_thresh]
                    print("syns:\n", )
                    return synfiltered[:min(self.number_of_synonyms,len(synfiltered))]
        return None
    
    def cluster_distance(self,verb):
        return vecs.distance(self.cluster_centroid(self.object_cluster(verb)),self.instance_centroid)

    def find_substitutes(self):
        syn = self.verb_synonyms()
        subs  = sorted([(s,self.cluster_distance(s)) for s in syn])
        self.substitutes = subs
        return subs 
    
    def substitute(self):
        if not self.substitutes:
            self.find_substitutes()
        return self.substitutes[0]



if __name__ ==  '__main__':
    #v = input("verb:") 
    #o = input("object:")
    ms = MetaphorSubstitute({
        "verb": 'shape',
        "obj":  'result',
        "object_cluster_size" : 10,
        "instance_object_weight": 5,
        "synonym_draw" : 10,
        "synonyms_file" : 'synonyms.txt',
        "search_left" : 0,
        "search_right": 5,
        "abstract_thresh" : 0.6,
        "min_MI" : 0.5,
        "number_of_synonyms" : 15
    })
     
    #ms.find_substitutes()
    #print("finally:\n",ms.substitutes)
