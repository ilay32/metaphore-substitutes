import math,pandas,sys,pickle,yaml,io,random
import numpy as np
from nltk.corpus import wordnet as wn
from dbutils import *
from prygress import progress

vecs = Vecs(DB('LSA-ENG'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lex = Lex(ngrams)

def preeval(fn):
    def inside(*args,**kwargs):
        if SingleWordData.empty(args[0].substitutes):
            print("You have to run get_substitutes first")
            return False
        if SingleWordData.empty(args[0].gold):
            print("there is no gold set to compare with")
            return False
        ans = fn(*args,**kwargs)
        return ans
    return inside

class MetaphorSubstitute:
    def __init__ (self,options):
        print("initializing",options['pred'],"--",options['noun'])
        for key in options.keys():
            self.__dict__[key] = options[key]
        self.substitutes = list()
        self.no_abst = set()
        self.no_vector = set()
        self.too_far = set()
        self.go = True
        self.classname = self.__class__.__name__
        self.instance_centroid = self.get_instance_centroid()
        if SingleWordData.empty(self.instance_centroid):
            self.go = False
    
    def __str__(self):
        return "predicate: {0}, object: {1}".format(self.pred,self.noun);

    
    @preeval
    def mrr(self):
        mrr = 0
        tst = [s[0] for s in self.substitutes]
        for t in self.gold:
            if t in tst:
                mrr += 1/(tst.index(t)+1)
        return mrr
    
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
        raw_cluster = self.noun_cluster(self.pred,'object')
        if SingleWordData.empty(raw_cluster):
            print("this instance has no cluster. calling it quits")
            return None 
        cent = self.cluster_centroid(raw_cluster)
        return Vecs.multiply(Vecs.addition(cent,Vecs.multiply(vecs.get(self.noun),self.instance_noun_weight)),1/2)
    

    def noun_cluster(self,pred,rel):
        print("fetching "+rel+"s set for "+pred)
        clust = list()
        added = cur =  0
        clusters = eval(self.classname).object_clusters if rel == 'object' else eval(self.classname).subject_clusters
        if not clusters.has(pred):
            print("can't find noun cluster for ",pred)
            return None
        clust_max = clusters.get(pred)
        while added < self.noun_cluster_size and cur < len(clust_max):
            lem = clust_max[cur] 
            if not SingleWordData.empty(lem):
                good = True
                if not vecs.has(lem):
                    print(lem+" has no vector representation. ignored.")
                    self.no_vector.add(lem)
                    good = False
                if not abst.has(lem) and good:
                    print(lem+" has no abstractness degree. ignored.")
                    self.no_abst.add(lem)
                    good = False
                if good:
                    clust.append(lem)
                    added += 1
            cur += 1
        if added == 0:
            print("all",len(clust_max)," found nouns are either not abstract enough or vectorless.",pred,"is ignored.")
            return None
        print("found:",printlist(clust,self.noun_cluster_size,True))
        return clust
    
    @hereiam
    def pred_synonyms_wordnet(self):
        print("getting synonyms")
        acc = list()
        syn = set()
        
        # synonyms
        s1 = [s for s in wn.synsets(self.pred) if self.wn_mark in s.name()]
        
        # add them
        acc += s1
        
        # loop synonyms to get heypernym sets *depth* steps up
        for s in s1:
            acc += s.closure(lambda x: x.hypernyms(), depth=2)        
        
        # loop immediate hypernyms and get all their pred hyponyms
        #for s in s1:
        #    for h in s.hypernyms():
        #        acc += [hypo for hypo in h.hyponyms() if ".v." in hypo.name()]
        
        # loop the accumulated sets and collect lemmas
        for s in acc:
            syn = syn.union(set([l.name() for l in s.lemmas() if l.name() != self.pred and vecs.has(l.name())]))
        syn = list(syn)
        syn.sort(key=lambda x: Vecs.distance(self.instance_centroid,vecs.get(x)),reverse=True)
        a = abst.get(self.pred)
        abstsyn = [s for s in syn if abst.get(s) > a]
        if not SingleWordData.empty(abstsyn):
            syn = abstsyn
        print("from wordnet:",printlist(syn,5,True))
        return syn

    def get_candidates(self):
        print("fetching candidate replacements for "+self.pred)
        cands = set()
        candidates = eval(self.classname).pred_candidates
        added = cur =  0
        if not candidates.has(self.noun):
            print("can't find typical preds for ",self.noun)
            return None
        cands_max = candidates.get(self.noun)
        added = cur = 0
        while added < self.number_of_candidates and cur < len(cands_max):
            while cands_max[cur] == self.pred:
                cur += 1
            candidate = cands_max[cur]
            if not vecs.has(candidate):
                print(candidate," has no vector representation. ignored.")
                self.no_vector.add(candidate)
            elif vecs.word_distance(candidate,self.pred) > self.candidate_sphere_radius:
                print(candidate," is too far from ",self.pred,". ignored")
                self.too_far.add(candidate)
            elif not abst.has(candidate):
                print(candidate," has no abstractness degree. ignored")
                self.no_abst.add(candidate)
            elif abst.get(candidate) > abst.get(self.pred):
                cands.add(candidate)
                added += 1
            cur += 1
        wn_syns = self.pred_synonyms_wordnet()
        # put the synonyms first while eliminating duplicates
        candlist = list(wn_syns) + [cand for cand in cands if cand not in wn_syns]
        if SingleWordData.empty(candlist):
            print("all ",len(cands_max)," candidates are either vectorless or too far from ",self.pred)
            return None
        print("found:",printlist(candlist,self.number_of_candidates,True))
        return candlist

    @hereiam
    def find_substitutes(self):
        candidates = self.get_candidates()
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                vc = self.noun_cluster(pred,'object')
                if not SingleWordData.empty(vc):
                    subs.append((pred,self.cluster_distance(vc)))
            subs = sorted(subs,key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs 
    
    def cluster_distance(self,cluster):
        cent = self.cluster_centroid(cluster)
        if cent:
            return Vecs.distance(cent,self.instance_centroid)
        else:
            return 0

    def substitute(self):
        if not self.substitutes:
            self.find_substitutes()
        return self.substitutes[0][0]

class AdjSubstitute(MetaphorSubstitute):
    object_clusters = AdjObjects(ngrams)
    pred_candidates = ObjectAdjs(ngrams) 
    def __init__(self,options):
        self.wn_mark = ".a."
        super(AdjSubstitute,self).__init__(options)

    @hereiam
    def neuman_eval(self):
        return int(self.substitute() == self.correct)

        # this is probably all wrong    
    def verb_synonyms_noam(self):
        identifier = self.noun+"_"+self.pred+"\t"
        with open(self.synonyms_file) as s:
            for line in s:
                if identifier in line:
                    synraw = line.partition(identifier)[2].split(',')
                    print(synraw[:15])
                    synfiltered = [w for w in synraw if lex.has(w) and 2 in lex.get(w) and abst.has(w) and abst.get(w) > self.abstract_thresh]
                    print("syns:\n", )
                    return synfiltered[:min(self.number_of_synonyms,len(synfiltered))]
        return None
    
    
    
    
