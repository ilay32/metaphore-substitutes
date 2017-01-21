import random,copy
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from dbutils import *
from prygress import progress
from ngraph import NeumanGraph
from sklearn.cluster import KMeans as km
params = yaml.load(open('params.yml'))
pairs = yaml.load(open('adj_pairs.yml'))
nouncats = pickle.load(open('nclass.pkl','rb'))

vectormodel = conf['vectormodel']
if vectormodel == "LSA":
    vecs = Vecs(DB('LSA-ENG'))
elif vectormodel == "SPVecs":
    vecs = SPVecs()
else:
    vecs = Word2Vec.load_word2vec_format(vectormodel,binary=True) 
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lemmatizer = WordNetLemmatizer()

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

def squeeze(words,lim):
    while len(words) > lim:
        out = vecs.doesnt_match(words)
        words.remove(out)
    return words

class MetaphorSubstitute:
    def __init__ (self,options):
        print("initializing",options['pred'],"--",options['noun'])
        for key in options.keys():
            self.__dict__[key] = options[key]
        self.noun = lemmatizer.lemmatize(self.noun)
        self.substitutes = list()
        self.no_abst = set()
        self.no_vector = set()
        self.too_far = set()
        self.classname = self.__class__.__name__
    
    def __str__(self):
        return  "{0} {1} top candidate: {2}".format(self.pred,self.noun,self.substitute())
    
    @preeval
    def mrr(self):
        mrr = 0
        tst = [s[0] for s in self.substitutes]
        for t in self.gold:
            if t in tst:
                mrr += 1/(tst.index(t)+1)
        return mrr
    
    def detect(self,adj):
        cparams = params['classifier']
        print("checking",adj,"literality")
        allnouns = eval(self.classname).object_clusters.get(adj).most_common(cparams['total'])
        concrete = [n[0] for n in sorted(allnouns,key=lambda x: abst.get(x[0]) if not SingleWordData.empty(x[0]) else 1)][:cparmas['kappa']]
        print("concrete:",printlist(concrete,5,True))
        categories = self.concrete_categories(concrete)
        for cat in categories:
            print(cat)
            if self.noun in nouncats[cat]:
                print(adj,"--",self.noun,"is literal. found category:",cat)
                return False #literal
        return True #metaphoric
        
    def concrete_categories(self,conclist):
        concrete_cats = list()
        for cat in nouncats:
            intersect = len(set(conclist).intersection(set(nouncats[cat])))
            if intersect >= cparams['cat_thresh']:
                concrete_cats.append(cat)
        if len(concrete_cats) > 0:
            print("found categories:",printlist(concrete_cats,len(concrete_cats),True))
        else:
            print("no category found")
        return concrete_cats

        
    def noun_cluster(self,pred,rel):
        print("fetching "+rel+"s set for "+pred)
        clust = list()
        added = cur =  0
        clusters = eval(self.classname).object_clusters if rel == 'object' else eval(self.classname).subject_clusters
        if not clusters.has(pred):
            print("can't find noun cluster for ",pred)
            return None
        clust = sorted([n[0] for n in clusters.get(pred).most_common() if n[0] in vecs and abst.has(n[0])],key=lambda x: abst.get(x),reverse=True)[:self.noun_cluster_size]
        if len(clust) == 0:
            print("all",len(clust)," found nouns are either not abstract enough or vectorless. only the instance noun will be used.")
        else:
            print("found:",printlist(clust,self.noun_cluster_size,True))
        #if(not self.noun in clust):
        #    clust.append(self.noun)
        return clust
    
    def wordnet_closure(word,pos):
        print("getting synonyms")
        syn = set()
        for s in wn.synsets(word,pos):
            syn = syn.union(set([l.name() for l in s.lemmas() ]))
        return list(syn)

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
    
    def candsrank(self,cand):
        return vecs.n_similarity([cand],self.get_synonyms()[:10])

    def find_substitutes(self):
        candidates = self.get_candidates()
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                subs.append((pred,self.candidate_rank(pred)))
            subs.sort(key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs 
    
    def cluster_distance(self,pred,cluster):
        cent = MetaphorSubstitute.cluster_centroid(cluster)
        if cent:
            return Vecs.vec_similarity(cent,self.instance_centroid)
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
    adjsyns = dict()
    
    def __init__(self,options):
        super(AdjSubstitute,self).__init__(options)
        self.wordnet_class = wn.ADJ
        self.nc = None
        self.strong_syns = list()
        self.type = None
    
    def neuman_eval(self):
        return int(self.substitute() == self.correct)
    
    def candidate_rank(self,cand):
        if cand not in vecs:
            return float('nan')
        s = self.get_syns()
        asyns = sorted(s,key=lambda x: abst.get(x),reverse=True)[:10]
        return vecs.n_similarity([cand],asyns)
    
    def wn_syns(word):
        ret = set()
        for snset in wn.synsets(word,wn.ADJ):
            for lem in snset.lemmas():
                if lem.name() in vecs:
                    ret.add(lem.name())
        return ret 
    
    def get_syns(self):
        if self.pred in AdjSubstitute.adjsyns:
            return AdjSubstitute.adjsyns[self.pred]
        else:
            csyns = set(self.coca_syns)
            wsyns = AdjSubstitute.wn_syns(self.pred)
            syns = list(csyns.union(wsyns))
            syns = [s for s in syns if s in vecs]
            AdjSubstitute.adjsyns[self.pred] = syns
            return syns
    
    def modifies_noun(self,adj):
        objs = AdjSubstitute.object_clusters.get(adj)
        if self.noun in objs:
            return objs.freq(self.noun)
        return 0
    
    def filter_antonyms(self,words):
        print("filtering antonyms")
        rem = list()
        for word in words:
            synsts = wn.synsets(word,wn.ADJ)
            syns = self.get_syns()[:10]
            for snset in synsts:
                for lem in snset.lemmas():
                    ants = lem.antonyms()
                    for ant in [a.name() for a in ants if a.name() in vecs]:
                        remove  = ""
                        add = ""
                        print("checking",ant,"against",word)
                        dis1 = vecs.n_similarity([word],syns)
                        dis2 = vecs.n_similarity([ant],syns)
                        if dis1 > dis2:
                            remove = ant
                        elif dis2 > dis1:
                            remove = word
                            add = ant
                        if add in vecs and add not in words:
                            print("adding",add)
                            #words.append(add)
                        if remove not in rem and remove in words:
                            print("removing",remove)
                            words.remove(remove)
                            rem.append(remove)
        return words

    def get_candidates(self):
        print("fetching candidate replacements for "+self.pred)
        cdatasource = eval(self.classname).pred_candidates
        if not cdatasource.has(self.noun):
            print("can't find typical preds for ",self.noun)
            return None
        cands_max = cdatasource.get(self.noun).most_common()
        cands = set()
        syns = self.get_syns()
        syns.sort(key=lambda x: self.modifies_noun(x),reverse=True)
        strong_syns = list()
        mat = list()
        good = list()
        for cand in cands_max:
            c = cand[0]
            if c in vecs and abst.has(c) and abst.get(c) > abst.get(self.pred):
                if c != self.pred and c in syns:
                    strong_syns.append(c)
                mat.append(vecs.get(c))
                good.append(cand)
        if len(strong_syns) > 0:
            self.strong_syns = strong_syns
        mat = np.array(mat)
        if len(cands_max) > 100 and False:
            k = km(n_clusters=round(len(cands_max)/50),random_state=0).fit(mat)
            hotclust = k.predict(vecs.get(self.pred).reshape(1,-1))
            cands = [cand for cand in good if k.predict(vecs.get(cand[0]).reshape(1,-1)) == hotclust]
            cands.sort(key=lambda x : x[1])
            #cands = squeeze(cands,15)
        else:
            cands = good
        cands = [cand[0] for cand in cands[:50]]
        if len(strong_syns) > 1:
            cands = set(cands[:5]).union(set(strong_syns))
            cands.discard(self.pred)
            ret = list(cands)
        else:  
            cands = set(cands[:15]).union(set(syns[:5]))
            cands.discard(self.pred)
            ret =  squeeze(list(cands),5)
        return ret

class NeumanAsIs(AdjSubstitute):
    """
    This class will only work with the LSA-ENG vector model
    """
    #store the filtered synonym lists and the concrete/abstract
    #vectors corresponding to each adjective
    adjdata = dict()
    
    # it's not clear what Neuman et al mean by "most connected".
    # the graph is directed, so is it in, out or all archs of a node.
    # so I just compute all three.
    ccriterion = 'all'
    
    def get_graph_data(adj,kind):
        """
        static helper that reads the graph previously
        computed according to Neuman et al description
        :param adj: the queried adjective
        :param: kind: concrete/abstract 
        :return the centroid of the vectors of the words pulled from the abstract/concrete graph of the current adjective
        """
        crit = NeumanAsIs.ccriterion
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,adj+"-"+kind+".pkl"),'rb'))
        proto_nouns = [n[0] for n in data[crit].most_common() if vecs.has(n[0])][:NeumanGraph.most_connected]
        return vecs.centroid(proto_nouns)
    

    def __init__(self,conf):
        super(NeumanAsIs,self).__init__(conf)
        self.params = params['neuman_exp2']
        self.classname += "ccriterion: "+NeumanAsIs.ccriterion
        if self.pred not in NeumanAsIs.adjdata:
            NeumanAsIs.adjdata[self.pred] = {
                'concrete_centroid' : NeumanAsIs.get_graph_data(self.pred,'concrete'),
                'abstract_centroid' : NeumanAsIs.get_graph_data(self.pred,'abstract'),
            }
    
    
    def get_synonyms(self):
        if 'syns' in NeumanAsIs.adjdata[self.pred]:
            return copy.copy(NeumanAsIs.adjdata[self.pred]['syns'])
        ret = list()
        simcut =  self.params['thetacut']
        n = self.params['thetanum']
        syns = [a for a in pairs[self.pred]['coca_syns'] if vecs.has(a)][:n]
        for s in syns:
            sim_abst = Vecs.vec_similarity(NeumanAsIs.adjdata[self.pred]['abstract_centroid'],vecs.get(s))
            sim_conc = Vecs.vec_similarity(NeumanAsIs.adjdata[self.pred]['concrete_centroid'],vecs.get(s))
            if sim_abst > simcut and sim_conc < simcut:
                ret.append(s)
            else:
                print(s,"ruled out by protolist")
        print("in total ruled out for", self.pred+":",len(syns) - len(ret))
        # this was not clear in the paper, I'm assuming it makes sense
        ret.append(self.pred)
        NeumanAsIs.adjdata[self.pred]['syns'] = ret
        return ret
    
    def get_candidates(self):
        return self.topfour
    
    def candidate_rank(self,cand):
        l = self.get_synonyms() + [self.noun]
        return vecs.n_similarity([cand],l)

class SimpleNeuman(AdjSubstitute):
    def candidate_rank(self,cand):
        syns = self.get_syns()
        if cand in vecs:
            return vecs.n_similarity([cand],syns)
        return float('nan') 
    
    def get_candidates(self):
        return self.topfour

    def get_syns(self):
        return self.coca_syns[:10]

class ASimpleNeuman(SimpleNeuman):
    def get_syns(self):
        syns = [s for s in self.coca_syns if s in vecs]
        return sorted(syns,key=lambda x: abst.get(x),reverse=True)[:10]

class SNwithNounData(SimpleNeuman):
    def candidate_rank(self,cand):
        frompred = super(SNwithNounData,self).candidate_rank(cand)
        candnouns = self.noun_cluster(cand,'object')
        fromnouns = vecs.n_similarity([self.noun],candnouns) if not SingleWordData.empty(candnouns) else 1
        return frompred*fromnouns
        

    
class AdjWithGraphProt(AdjSubstitute):
    def noun_cluster(self,pred,rel):
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,pred+"-abstract.pkl"),'rb'))
        proto_nouns = [n[0] for n in data['out'].most_common() if vecs.has(n[0])][:self.noun_cluster_size]
        print("abstract objects for",pred+":",printlist(proto_nouns,10,True))
        return proto_nouns 
    
    def get_candidates(self):
        return self.topfour

