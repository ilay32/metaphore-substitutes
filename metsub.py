from nltk.corpus import wordnet as wn
from dbutils import *
from prygress import progress
from ngraph import NeumanGraph
import random
vecs = Vecs(DB('LSA-ENG'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lex = Lex(ngrams)
nouns = NounNoun(ngrams)

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
        ret =  "{0} {1}".format(self.pred,self.noun)
        if self.go:
            ret += " top candidate: "+self.substitute()
        else:
            ret += " noun not known"
        return ret
    
    @preeval
    def mrr(self):
        mrr = 0
        tst = [s[0] for s in self.substitutes]
        for t in self.gold:
            if t in tst:
                mrr += 1/(tst.index(t)+1)
        return mrr
    
    def cluster_centroid(cluster):
        if SingleWordData.empty(cluster):
            print("empty cluster")
            return None
        cluster = [w for w in cluster if vecs.has(w)]
        if SingleWordData.empty(cluster):
            print("no vector for any of the words in this cluster")
            return None
        print("computing cluster centroid for ",printlist(cluster,7,True))
        acc = vecs.get(cluster[0])
        added = 1
        for word in cluster[1:]:
            if SingleWordData.empty(vecs.get(word)): 
                print("nan in vector", word)
            else:
                acc = Vecs.addition(acc,vecs.get(word))
                added += 1
        return Vecs.multiply(acc,1/added)

    def get_instance_centroid(self):
        print("computing instance centroid")
        raw_cluster = self.noun_cluster(self.pred,'object')
        if SingleWordData.empty(raw_cluster):
            print("this instance has no cluster. calling it quits")
            return None 
        cent = MetaphorSubstitute.cluster_centroid(raw_cluster)
        instance_noun_vec = Vecs.multiply(vecs.get(self.noun),self.instance_noun_weight)
        instance_pred_vec = vecs.get(self.pred)
        syncent = MetaphorSubstitute.cluster_centroid(self.get_synonyms())
        return Vecs.subtract(vecs.get(self.pred),vecs.get(self.noun))
         

    def noun_cluster(self,pred,rel):
        print("fetching "+rel+"s set for "+pred)
        clust = list()
        added = cur =  0
        clusters = eval(self.classname).object_clusters if rel == 'object' else eval(self.classname).subject_clusters
        if not clusters.has(pred):
            print("can't find noun cluster for ",pred)
            return None
        clust = sorted([n for n in clusters.get(pred) if vecs.has(n) and abst.has(n)],key=lambda x: abst.get(x),reverse=True)[:self.noun_cluster_size]
        if len(clust) == 0:
            print("all",len(clust)," found nouns are either not abstract enough or vectorless. only the instance noun will be used.")
        else:
            print("found:",printlist(clust,self.noun_cluster_size,True))
        #if(not self.noun in clust):
        #    clust.append(self.noun)
        return clust
    
    def wordnet_closure(word,pos):
        print("getting synonyms")
        acc = list()
        syn = set()
        
        s1 = wn.synsets(word,pos)
        
        acc += s1
        # loop synonyms to get heypernym sets *depth* steps up
        for s in s1:
            acc += s.closure(lambda x: x.hypernyms(), depth=2)        
        
        # loop immediate hypernyms and get all their pred hyponyms
        #for s in s1:
        #    for h in s.hypernyms():
        #        acc += h
        #        acc += h.hyponyms(pos)
        #
        #loop the accumulated sets and collect lemmas
        for s in acc:
            syn = syn.union(set([l.name() for l in s.lemmas() if l.name() != word]))
        #a = abst.get(self.pred)
        #abstsyn = [s for s in syn if abst.get(s) > a]
        #if not SingleWordData.empty(abstsyn):
        #    syn = abstsyn
        print("from wordnet:",printlist(syn,10,True))
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
        wn_syns = MetaphorSubstitute.wordnet_closure(self.pred,self.wordnet_class)
        # put the synonyms first while eliminating duplicates
        candlist = list(wn_syns) + [cand for cand in cands if cand not in wn_syns]
        if SingleWordData.empty(candlist):
            print("all ",len(cands_max)," candidates are either vectorless or too far from ",self.pred)
            return None
        print("found:",printlist(candlist,self.number_of_candidates,True))
        return candlist

    def find_substitutes(self):
        candidates = self.get_candidates()
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                vc = self.noun_cluster(pred,'object')
                if not SingleWordData.empty(vc):
                    subs.append((pred,self.cluster_distance(pred,vc)))
            subs = sorted(subs,key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs 
    
    def cluster_distance(self,pred,cluster):
        cent = MetaphorSubstitute.cluster_centroid(cluster)
        if cent:
            return Vecs.distance(cent,self.instance_centroid)
        else:
            return 0

    def substitute(self):
        if not self.substitutes:
            self.find_substitutes()
        return self.substitutes[0][0]


###################################################
# overriding the generic metaphore substitute class
###################################################
class AdjSubstitute(MetaphorSubstitute):
    object_clusters = AdjObjects(ngrams)
    pred_candidates = ObjectAdjs(ngrams) 
    def __init__(self,options):
        self.wordnet_class = wn.ADJ
        self.nc = None
        super(AdjSubstitute,self).__init__(options)
    
        
    def noun_cluster_bynoun(self,pred,rel) :
        if self.nc:
            return self.nc
        else:
            clust = [n for n in nouns.get(self.noun) if abst.has(n) and vecs.has(n)][:100]
            clust.sort(key=lambda x : abst.get(x), reverse=True)
            nc = MetaphorSubstitute.cluster_centroid(clust[:self.noun_cluster_size])
            self.nc = nc
            return nc

    def neuman_eval(self):
        return int(self.substitute() == self.correct)
    
    def get_synonyms(self):
        return sorted(self.coca_syns,key=lambda x: abs(abst.get(x) - abst.get(self.pred)))[:20]

    def get_candidates(self):
        raw = [a for a in AdjSubstitute.pred_candidates.get(self.noun) if abst.has(a) and vecs.has(a) and abst.get(a) > abst.get(self.pred)]
        ret = raw[:min(len(raw), 20)]
        #random.shuffle(ret)
        #while self.correct in ret[:3]:
            #random.shuffle(ret)
        #ret = set(ret).union(set(sorted([a for a in self.coca_syns if abst.has(a) and vecs.has(a)][:max(nc,15)],key=lambda x: abst.get(x),reverse=True)[:nc]))
        #ret = ret[:3]
        #ret.append(self.correct)
        print("candidates:",printlist(ret,10,True))
        return ret 
    
    def get_instance_centroid(self) :
        #MetaphorSubstitute.cluster_centroid(self.get_synonyms())
        vp = vecs.get(self.pred)
        if vecs.has(self.noun):
            ret = Vecs.subtract(vp,vecs.get(self.noun))
        else:
            serrogate = MetaphorSubstitute.cluster_centroid(self.noun_cluster(self.pred,'object'))            
            ret = Vecs.subtract(vp,serrogate)
        return ret
    
    def cluster_distance(self,pred,clust):
        cent = MetaphorSubstitute.cluster_centroid(clust)
        d = Vecs.subtract(vecs.get(pred),cent)
        return Vecs.distance(d,self.instance_centroid)


class SimpleNeuman(AdjSubstitute):
    def find_substitutes(self):
        candidates = self.get_candidates()
        ic = self.get_instance_centroid()
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                subs.append((pred,Vecs.distance(vecs.get(pred),ic)))
            subs = sorted(subs,key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs 
        
    def get_instance_centroid(self):
        return MetaphorSubstitute.cluster_centroid(self.get_synonyms()[:10])

class AdjWithGraphProt(AdjSubstitute):
    def noun_cluster(self,pred,rel):
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,pred+"-abstract.pkl"),'rb'))
        proto_nouns = [n[0] for n in data['out'].most_common() if vecs.has(n[0])][:self.noun_cluster_size]
        print("abstract objects for",pred+":",printlist(proto_nouns,10,True))
        return proto_nouns 
    
    def get_candidates(self):
        return self.topfour

