# main frame of the program #
import math,pandas,sys,pickle,yaml
import numpy as np
from nltk.corpus import wordnet as wn
from pairs import pairs
from dbutils import *

conf = yaml.load(open('params.yml').read())
vecs = Vecs(DB('LSA-ENG'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lex = Lex(ngrams)
clusters = VerbObjects(ngrams)
candidates = ObjectVerbs(ngrams) 
class MetaphorSubstitute:
    def __init__ (self,options):
        print("initializing "+options['verb']+"--"+options['obj'])
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
        added = cur =  0
        good = True
        if not clusters.has(verb):
            print("can't find object cluster for ",verb)
            return None
        clust_max = clusters.get(verb)
        while added < self.object_cluster_size and cur < len(clust_max):
            lem = clust_max[cur] 
            if not vecs.has(lem):
                print(lem+" has no vector representation. ignored.")
                good = False
            if not abst.has(lem):
                print(lem+" has no abstractness degree. ignored.")
                good = False
            if good:
                clust.append(lem)
                added += 1
            cur += 1
        if added == 0:
            print("all ",len(clust_max)," found nouns are either not abstract enough or vectorless. ",verb," is ignored.")
            return None
        print("found: ",printlist(clust,self.object_cluster_size,True))
        return clust
    
    def cluster_centroid(self,cluster):
        if SingleWordData.empty(cluster):
            print("empty cluster")
            return None
        print("computing cluster centroid for ",printlist(cluster,7,True))
        added = 0
        acc = vecs.get(cluster[0])
        for word in cluster[1:]:
            if vecs.has(word):
                if SingleWordData.empty(vecs.get(word)): 
                    print("nan in vector", word)
                acc = Vecs.addition(acc,vecs.get(word))
                added += 1
        if added > 0:
            return Vecs.multiply(acc,1/added)
        else:
            return None

    def get_instance_centroid(self):
        print("computing instance centroid")
        raw_cluster = self.object_cluster(self.verb)
        if SingleWordData.empty(raw_cluster):
            print("this instance has no cluster. calling it quits")
            return None 
        cent = self.cluster_centroid(raw_cluster)
        return Vecs.multiply(Vecs.addition(cent,Vecs.multiply(vecs.get(self.obj),self.instance_object_weight)),1/2)
    
    # this is probably all wrong    
    def verb_synonyms_noam(self):
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
    
    def verb_synonyms_wordnet(self):
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
        print("found",printlist(syn,5,True))
        return set(syn)

    def get_candidates(self):
        print("fetching candidate replacements for "+self.verb)
        cands = set()
        added = cur =  0
        if not candidates.has(self.obj):
            print("can't find typical verbs for ",self.obj)
            return None
        cands_max = candidates.get(self.obj)
        added = cur = 0
        while added < self.number_of_candidates and cur < len(cands_max):
            candidate = cands_max[cur]
            if not vecs.has(candidate):
                print(candidate," has no vector representation. ignored.")
            elif vecs.word_distance(candidate,self.verb) > self.candidate_sphere_radius:
                print(candidate," is too far from ",self.verb,". ignored")
            else:
                cands.add(candidate)
                added += 1
            cur += 1
        cands = list(cands.union(self.verb_synonyms_wordnet())) 
        if SingleWordData.empty(cands):
            print("all ",len(cands_max)," candidates are either vectorless or too far from ",self.verb)
            return None
        print("found:",printlist(cands,max(self.number_of_candidates,25),True))
        return cands

    def cluster_distance(self,objects_cluster):
        cent = self.cluster_centroid(objects_cluster)
        if cent:
            return Vecs.distance(cent,self.instance_centroid)
        else:
            return 0

    def find_substitutes(self):
        candidates = self.get_candidates()
        subs = list()
        for verb in candidates:
            vc = self.object_cluster(verb)
            if not SingleWordData.empty(vc):
                subs.append((verb,self.cluster_distance(vc)))
        subs = sorted(subs,key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs 
    
    def substitute(self):
        if not self.substitutes:
            self.find_substitutes()
        return self.substitutes[0]



if __name__ ==  '__main__':
    #v = input("verb:") 
    #o = input("object:")
    limit = sys.argv[1] if len(sys.argv) > 1 else len(pairs)
    rundata = list()
    for i in range(int(limit)):
        o,v = pairs[i]
        conf.update({'verb':v,'obj':o})
        ms = MetaphorSubstitute(conf)
        if ms.go:
            subs = ms.find_substitutes()
            d = {"pair": v +" "+o, "substitutes" : subs}
            rundata.append(d)
        print(ms," done","\n====================\n")
    rundata = pandas.DataFrame(rundata)
    print("wrapping up and saving stuff")
    vecs.destroy()
    ngrams.destroy()
    abst.destroy()
    lex.destroy()
    clusters.destroy()
    candidates.destroy()     
    print("\n\t\tHura!\n")


