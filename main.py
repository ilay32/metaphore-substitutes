# main frame of the program #
import math,pandas,sys
import numpy as np
from nltk.corpus import wordnet as wn
from pairs import pairs
from dbutils import *
vecs = Vecs(DB('LSA-ENG'))
Ngrams = DB('NgramsCOCAAS')
abst = Abst(Ngrams)
lex = Lex(DB('NgramsCOCAAS'))

class MetaphorSubstitute:
    def __init__ (self,options):
        print("initializing "+options['verb']+", "+options['obj'])
        for key in options.keys():
            self.__dict__[key] = options[key]
        self.instance_centroid = self.get_instance_centroid()
        self.substitutes = list()
        self.go = True
        if SingleWordData.empty(self.instance_centroid):
            self.go = False
    
    def __str__(self):
        return "verb:"+self.verb+", object:"+self.obj;
    
    # get n nouns, up to k words to the right of the verb,
    # in this version,simple filtering by the instance abstractness
    # threshold
    def object_cluster(self,verb):
        print("fetching objects set for "+verb)
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
        cur = 0
        try:
            rows = q.fetchall()
            while added < min(self.object_cluster_size,len(rows) - 1):
                lem = rows[cur][10] # take the lemma in this case ?
                if(vecs.has(lem)):
                    clust.append(lem)
                    added += 1
                cur += 1
        except:
            print(verb+" can't find noun cluster. ignored")
        return clust

    def cluster_centroid(self,cluster):
        print("computing cluster centroid for ",cluster[:5],"...")
        added = 0
        acc = vecs.get(cluster[0])
        for word in cluster[1:]:
            if vecs.has(word):
                if SingleWordData.empty(vecs.get(word)): 
                    print("nan in vector", word)
                acc = Vecs.addition(acc,vecs.get(word))
                added += 1
        return Vecs.multiply(acc,1/added)
    
    def get_instance_centroid(self):
        print("computing instance centroid")
        raw_cluster = self.object_cluster(self.verb)
        if SingleWordData.empty(raw_cluster):
            print("this instance has no cluster. calling it quits")
            return None 
        cent = self.cluster_centroid(raw_cluster)
        return Vecs.multiply(Vecs.addition(cent,Vecs.multiply(vecs.get(self.obj),self.instance_object_weight)),1/2)
    
    # this is probably all wrong    
    #def verb_synonyms(self):
    #    identifier = self.obj+"_"+self.verb+"\t"
    #    with open(self.synonyms_file) as s:
    #        for line in s:
    #            if identifier in line:
    #                synraw = line.partition(identifier)[2].split(',')
    #                print(synraw[:15])
    #                synfiltered = [w for w in synraw if lex.has(w) and 2 in lex.get(w) and abst.has(w) and abst.get(w) > self.abstract_thresh]
    #                print("syns:\n", )
    #                return synfiltered[:min(self.number_of_synonyms,len(synfiltered))]
    #    return None
    def verb_synonyms(self):
        print("getting synonyms:")
        synsets = wn.synsets(self.verb)
        syn = list()
        # make sure they're verbs
        synsets = [s for s in synsets if ".v." in s.name()]
        # loop once to get heypernym sets
        for s in synsets:
            synsets += s.hypernyms()
        # loop again to collect lemmas
        for s in synsets:
            syn += [l.name() for l in s.lemmas() if l.name() != self.verb]
        print("found",syn[:5],"...")
        return list(set(syn[:min(self.number_of_synonyms,len(syn))]))

    def cluster_distance(self,objects_cluster):
        return Vecs.distance(self.cluster_centroid(objects_cluster),self.instance_centroid)

    def find_substitutes(self):
        syn = self.verb_synonyms()
        subs = list()
        for verb in syn:
            vc = self.object_cluster(verb)
            if not SingleWordData.empty(vc):
                subs.append((verb,self.cluster_distance(vc)))
            subs = sorted(subs)
        self.substitutes = subs
        return subs 
    
    def substitute(self):
        if not self.substitutes:
            self.find_substitutes()
        return self.substitutes[0]



if __name__ ==  '__main__':
    #v = input("verb:") 
    #o = input("object:")
    limit = sys.argv[1]
    if not limit:
        limit = len(pairs)
    rundata = list()
    for i in range(int(limit)):
        o,v = pairs[i]
        ms = MetaphorSubstitute({
            "verb": v,
            "obj":  o,
            "object_cluster_size" : 10,
            "instance_object_weight": 5,
            "synonym_draw" : 10,
            "synonyms_file" : 'synonyms.txt',
            "search_left" : 0,
            "search_right": 5,
            "abstract_thresh" : 0.6,
            "min_MI" : 0.1,
            "number_of_synonyms" : 15
        })
        if ms.go:
            subs = ms.find_substitutes()
            d = {"pair": v +"(v), "+o+ "(n)", "substitutes" : subs}
            rundata.append(d)
        print(ms," done","\n====================\n")
    rundata = pandas.DataFrame(rundata)
    print("\n\t\tHura!\n")




