# main frame of the program #
import math,pandas,sys,pickle,yaml,io,random
import numpy as np
from nltk.corpus import wordnet as wn
from dbutils import *

conf = yaml.load(open('params.yml').read())
vecs = Vecs(DB('LSA-ENG'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lex = Lex(ngrams)
object_clusters = VerbObjects(ngrams)
subject_clusters = VerbSubjects(ngrams)
object_candidates = ObjectVerbs(ngrams) 
subject_candidates = SubjectVerbs(ngrams) 

class MetaphorSubstitute:
    def __init__ (self,options):
        print("initializing "+options['verb']+"--"+options['noun'])
        for key in options.keys():
            self.__dict__[key] = options[key]
        self.substitutes = list()
        self.no_abst = set()
        self.no_vector = set()
        self.ignored_verbs = set()
        self.go = True
        self.instance_centroid = self.get_instance_centroid()
        if SingleWordData.empty(self.instance_centroid):
            self.go = False
    
    def evaluate(self):
        if SingleWordData.empty(self.substitutes):
            print("You have to run get_substitutes first")
            return False
        if SingleWordData.empty(self.gold):
            return False
        mrr = 0
        tst = self.substitutes
        gld = self.gold
        l = min(len(tst),len(gld))
        for t in tst[:l]:
            if t in gld:
                mrr += 1/gld.index(t)+1
        return mrr/l
            

    def __str__(self):
        return "verb:"+self.verb+", "+self.rel+":"+self.noun;
    
    # get n nouns, up to k words to the right of the verb,
    # in this version,simple filtering by the instance abstractness
    # threshold
    @hereiam
    def noun_cluster(self,verb):
        print("fetching "+self.rel+"s set for "+verb)
        clust = list()
        added = cur =  0
        clusters = object_clusters if self.rel == 'object' else subject_clusters
        if not clusters.has(verb):
            print("can't find noun cluster for ",verb)
            return None
        clust_max = clusters.get(verb)
        print("looping nouns",self.noun_cluster_size,len(clust_max)) 
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
            print("all ",len(clust_max)," found nouns are either not abstract enough or vectorless. ",verb," is ignored.")
            self.ignored_verbs.add(verb)
            return None
        print("found: ",printlist(clust,self.noun_cluster_size,True))
        return clust
    
    @hereiam
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

    @hereiam
    def get_instance_centroid(self):
        print("computing instance centroid")
        raw_cluster = self.noun_cluster(self.verb)
        if SingleWordData.empty(raw_cluster):
            print("this instance has no cluster. calling it quits")
            return None 
        cent = self.cluster_centroid(raw_cluster)
        return Vecs.multiply(Vecs.addition(cent,Vecs.multiply(vecs.get(self.noun),self.instance_noun_weight)),1/2)
    
    # this is probably all wrong    
    def verb_synonyms_noam(self):
        identifier = self.noun+"_"+self.verb+"\t"
        with open(self.synonyms_file) as s:
            for line in s:
                if identifier in line:
                    synraw = line.partition(identifier)[2].split(',')
                    print(synraw[:15])
                    synfiltered = [w for w in synraw if lex.has(w) and 2 in lex.get(w) and abst.has(w) and abst.get(w) > self.abstract_thresh]
                    print("syns:\n", )
                    return synfiltered[:min(self.number_of_synonyms,len(synfiltered))]
        return None
    
    @hereiam
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
        candidates = object_candidates if self.rel == 'object' else subject_candidates
        if not candidates.has(self.noun):
            print("can't find typical verbs for ",self.noun)
            return None
        cands_max = candidates.get(self.noun)
        added = cur = 0
        while added < self.number_of_candidates and cur < len(cands_max):
            candidate = cands_max[cur]
            if not vecs.has(candidate):
                print(candidate," has no vector representation. ignored.")
                self.no_vector.add(candidate)
            elif vecs.word_distance(candidate,self.verb) > self.candidate_sphere_radius:
                print(candidate," is too far from ",self.verb,". ignored")
                self.ignored_verbs.add(candidate)
            elif not abst.has(candidate):
                print(candidate," has no abstractness degree. ignored")
                self.no_abst.add(candidate)
            elif abst.get(candidate) > self.abstract_thresh:
                cands.add(candidate)
                added += 1
            cur += 1
        wn_syns = self.verb_synonyms_wordnet()
        # put the synonyms first while eliminating duplicates
        candlist = list(wn_syns) + [cand for cand in cands if cand not in wn_syns]
        if SingleWordData.empty(candlist):
            print("all ",len(cands_max)," candidates are either vectorless or too far from ",self.verb)
            return None
        print("found:",printlist(candlist,self.number_of_candidates,True))
        return candlist

    @hereiam
    def cluster_distance(self,cluster):
        cent = self.cluster_centroid(cluster)
        if cent:
            return Vecs.distance(cent,self.instance_centroid)
        else:
            return 0

    @hereiam
    def find_substitutes(self):
        candidates = self.get_candidates()
        subs = list()
        if not SingleWordData.empty(candidates):
            for verb in candidates:
                vc = self.noun_cluster(verb)
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
    pairs = yaml.load(open('pairs.yml'))
    verb_object = sorted(list(pairs['verb_object'].values()),key = lambda x: x['verb'])
    verb_subject = sorted(list(pairs['verb_subject'].values()),key = lambda x: x['verb'])
    print("verb - object pairs:")
    for i,t in enumerate(verb_object):
        print(str(i+1)+".",t['verb'],",",t['noun'])
    print("verb - subject pairs:")
    for i,t in enumerate(verb_subject):
        print(str(i+len(verb_object)+1)+".",t['verb'],",",t['noun'])
    print("modes:")
    print("1. number -- pick a pair of the above")
    print("2. rnd n randomly pick n of the (object,verb) kind")
    print("3. all")
    print("4. only object,verb")
    print("5. only subject,verb")
    
    run = list()
    mode = ""
    while mode not in ["1","2","3","4","5"]:
        mode = input("choose mode: ")
    if mode == "1":
        pair = int(input("pick a pair by it's number on the list: ")) - 1
        if pair < len(verb_object):
            run.append(verb_object[pair])
        else:
            run.append(verb_subject[pair - len(verb_object)])
    
    if mode == "2":
        random.shuffle(verb_object)
        num = int(input("how many: "))
        run = verb_object[:num]
    
    if mode == "3":
        run = verb_object + verb_subject
    
    if mode == "4":
        run = verb_object

    if mode == "5":
        run = verb_subject 
    
    rundata = list()
    note = input("add a note about this run:")
    for p in run:
        conf.update(p)
        ms = MetaphorSubstitute(conf)
        if ms.go:
            subs = ms.find_substitutes()
            d = {
                "verb": p['verb'],
                "noun" : p['noun'],
                "rel" : p['rel'],
                "substitutes" : subs, 
                "no_vector" : ms.no_vector, 
                "no_abst" : ms.no_abst,
                "ignored" : ms.ignored_verbs,
                "score" : ms.evaluate()
            }
            rundata.append(d)
        print(ms," done","\n====================\n")
    rundata = RunData(pandas.DataFrame(rundata),note)
    print("wrapping up and saving stuff")
    rundata.save()
    vecs.destroy()
    ngrams.destroy()
    abst.destroy()
    lex.destroy()
    object_clusters.destroy()
    subject_clusters.destroy()
    object_candidates.destroy()     
    subject_candidates.destroy()     
    print("\n\t\tHura!\n")


