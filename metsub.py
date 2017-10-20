#   imports
#--------------------
import random,copy,re,nltk,string,time,numpy,subprocess,regex
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models  import KeyedVectors
from dbutils import *
from prygress import progress
from ngraph import NeumanGraph
from sklearn.cluster import KMeans as km
from scipy.stats import spearmanr,entropy,percentileofscore
from lxml import etree


#   globals
#----------------------
ggrams = Erlangen()
ggrams2 = Erlangen2()
params = yaml.load(open('params.yml'))
pairs = yaml.load(open(params['pairs_file']))
nouncats = pickle.load(open('nclass.pkl','rb'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lemmatizer = WordNetLemmatizer()
vectormodel = conf['vectormodel']
semevaltest = etree.parse('../semeval/lexsub_test.xml')
#coinco = etree.parse('../coinco/coinco.xml')
#cropper = etree.XSLT('../coinco/adjoptions.xsl')
if vectormodel == "LSA":
    vecs = Vecs(DB('LSA-ENG'))
elif vectormodel == "SPVecs":
    vecs = SPVecs()
    vecs.has = lambda x: x in vecs
    
else:
    vecs = KeyedVectors.load_word2vec_format(vectormodel,binary=True)
    vecs.has = lambda x: x in vecs
    vecs.get = lambda x: vecs[x]
    vecs.centroid = lambda l: vecs[vecs.most_similar(l,topn=1)[0][0]]



#   helpers
#-----------------
def antonyms(target_word,pos=None):
    synsets = wn.synsets(target_word,pos)
    ants = set()
    for s in synsets:
        for l in s.lemmas():
            ants = ants.union(set([ant.name() for ant in l.antonyms()]))
    return ants

def clearvecs(l,ind=None):
    ret = list()
    for x in l:
        y = x[ind] if ind is not None else x
        if y in vecs:
            ret.append(y)
        else:
            ly = lemmatizer.lemmatize(y)
    return ret

def overlap(a,b):
    if isinstance(a,list):
        a = set(a)
    if isinstance(b,list):
        b = set(b)
    return len(a.intersection(b))/min(len(a),len(b))

def squeeze(words,lim):
    while len(words) > lim:
        out = vecs.doesnt_match(words)
        words.remove(out)
    return words

#   decorators
#-----------------
def preeval(fn):
    def inside(*args,**kwargs):
        if SingleWordData.empty(args[0].substitutes):
            print("you have to run find_substitutes first ---",args[0].pred,args[0].noun)
            return False
        if SingleWordData.empty(args[0].gold):
            print("there is no gold set to compare with --- ",args[0].pred,args[0].noun)
            return False
        ans = fn(*args,**kwargs)
        return ans
    return inside

class MetaphorSubstitute:
    """
    generic 'get candidates, rate candidates' paraphraser
    """
    argpatt = re.compile("^(\w+)(\([,\d]+\))?$")
    
    def __init__ (self,options):
        print("initializing",options['pred'],"--",options['noun'])
        
        #incroporate all options as self properties
        for key in options.keys():
            self.__dict__[key] = options[key]
               
        #lemmatize the instance noun (?)
        #self.noun = lemmatizer.lemmatize(self.noun)
        self.lnoun = lemmatizer.lemmatize(self.noun)
        self.substitutes = list()
        self.no_abst = set()
        self.no_vector = set()
        self.too_far = set()
        self.coverage = 0
        self.classname = self.__class__.__name__
        if not self.dry_run:
            self.resolve_methods()    
    
    def resolve_methods(self):
        self.get_candidates = eval('self.'+self.methods['candidates'])
        self.candidate_rank = eval('self.'+self.methods['rating'])
           

    def __str__(self):
        return  "{0} {1} top candidate: {2}".format(self.pred,self.noun,self.substitute())
    
    #*************************  
    #   essence
    #*************************
    def find_substitutes(self):
        """
        Shell for the basic rating algorithm

        *get_candidates* and *rate candidates*
        are specified as parameters to be sorted out by *resolve_methods*.
        this allows for maximal flexibility without a deep and branchy
        inheritance tree.
        """
        ants = self.pred_antonyms()
        candidates = [c for c in self.get_candidates() if c not in ants]
        
        if SingleWordData.empty(candidates):
            return []
        self.coverage = overlap(candidates,self.gold)
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                subs.append((pred,self.candidate_rank(pred)))
            subs.sort(key=lambda t: t[1],reverse=True)
        self.substitutes = subs
        return subs

    def substitute(self):
        if self.substitutes is None:
            self.find_substitutes()
        if len(self.substitutes) > 0:
            return self.substitutes[0][0]
        else:
            return 'none'
    
    def export_data(self):
        subs = self.find_substitutes()
        return {
            "pred": self.pred,
            "noun" : self.noun,
            "substitutes" : subs,
            "neuman_score" : self.neuman_eval(),
            "neuman_correct" : pairs[self.pred]['with'][self.noun].get('neuman_correct'),
            "avprec" : self.AP(),
            "strictprec" : self.strictP(),
            "gap" : self.GAP(),
            "spearmanr" : self.spear(),
            "overlap" : self.overlap(),
            "coverage" : self.coverage,
            "top_in_gold" : self.lenient_acc(),
            "none_in_gold" : self.complete_miss(),
            "oot" : self.oot(),
            "best" : self.best(),
            "cands_size" : len(subs),
            "semid" : self.semid,
            "KL" : self.KL()#,
            #"isnovel" : 1 -  self.self_rrep()
        }
    #******************************* 
    #   evaluation methods           
    #*******************************
    @preeval
    def oot(self):
        g,t = self.intersection_rates()
        return sum(g)/sum(self.gold_rates)

    @preeval
    def best(self,cutoff=1):
        g,t = self.intersection_rates(cutoff)
        if len(g) == 0:
            return 0
        return sum(g)/sum(self.gold_rates)*cutoff

    @preeval
    def strictP(self):
        r = 0
        sampsize = min(len(self.gold),len(self.substitutes))
        tst = [s[0] for s in self.substitutes[:sampsize]]
        gld = self.gold
        for i,t in enumerate(tst,1):
            rank = 1/i  #1/sampsize to 1
            worth = len(gld) - gld.index(t)  if t in gld else 0 #0 .. len(gld) 
            r += (rank * worth)/len(gld) #0 .. 1
        return r/sampsize # 0 .. 1
    
    @preeval
    def AP(self):
        ans = 0
        lg = len(self.gold)
        tst = [s[0] for s in self.substitutes]
        for i,t in enumerate(tst,1):
            ans += len([s for s in tst[:i] if s in self.gold])/i
        return ans/len(tst)
    
    @preeval
    def GAP(self):
        keyed = dict(zip(self.gold,self.gold_rates))         
        tst = [s[0] for s in self.substitutes[:10]]
        tstrates = [keyed.get(s) or 0 for s in tst]
        numer = 0
        for i,t in enumerate(tstrates,1):
            if t != 0:
                numer += sum([r for r in tstrates[:i]])/i
        return numer/sum(self.gold_rates)

    @preeval
    def spear(self):
        acc = 0
        tranks = list()
        # this assumes the gold *ranks* are unique and the list is sorted
        for i,s in enumerate(self.substitutes):
            if s[0] in self.gold:
                gi = self.gold.index(s[0])
                tranks.append(gi)
                acc += (gi - i)**2
        c = len(tranks)
        if c > 2:
            return 1 - (6 * acc)/(c**3 - c)
        elif c > 1:
            return np.corrcoef(np.arange(1,c+1),tranks)[0,1]
        elif c == 1:
            return 'undefined'
        else:
            return -1
         
    @preeval
    def overlap(self):
        return overlap(self.gold,[ w[0] for w in self.substitutes])
    
    @preeval
    def lenient_acc(self):
        return int(self.substitute() in self.gold)

    @preeval
    def complete_miss(self):
        return int(len([s for s in self.substitutes if s[0] in self.gold]) == 0)
    
    @preeval 
    def neuman_eval(self):
        return int(self.substitute() == self.neuman_correct)

    @preeval
    def corrcoef(self):
        g,t = intersection_rates(gld,gldrnks,tst,tstrnks)
        if len(g) == 1:
            return 'undefined'
        else:
            g = np.array(g)
            t = np.array(t)
            if g.std() == 0 or t.std() == 0:
                return 0
        return np.corrcoef(g,t)[0,1]
    
    @preeval 
    def KL(self):
        return entropy(*self.intersection_rates())


    #********************** 
    #   rating methods      
    #**********************
      

    #   miscelenia
    #----------------------
    def isnovel(self,spread=1,n=20):
        #wnsyns = set(self.wncands(spread)())
        #cocsyns = self.cocands(c)()
        syns = set(self.roget_syns)
        mods = set(self.ngramcands(n)())
        return len(syns.intersection(mods)) == 0


    def erlangen_swaps(c):
       c = regex.sub(r'\p{P}', ' PUN ',c)
       c = re.sub(r'[0-9]+',' NUM ',c)
       return c

    @preeval
    def intersection_rates(self,limit=10):
        granks = list()
        tranks = list()
        l = len(self.substitutes) if limit is None else limit
        for s in self.substitutes[:l]:
            if s[0] in self.gold:
                granks.append(self.gold_rates[self.gold.index(s[0])])
                tranks.append(s[1])
        return granks,tranks
    
    def noun_cluster(self,pred,rel):
        print("fetching "+rel+"s set for "+pred)
        clust = list()
        added = cur =  0
        clusters = eval(self.classname).object_clusters if rel == 'object' else eval(self.classname).subject_clusters
        if not clusters.has(pred):
            print("can't find noun cluster for ",pred)
            return None
        clust = sorted([n[0] for n in clusters.get(pred).most_common() if n[0] in vecs and abst.has(n[0])],key=lambda x: abst.get(x),reverse=True)
        if len(clust) == 0: 
            print("all",len(clust)," found nouns are either not abstract enough or vectorless. only the instance noun will be used.")
        else:
            print("found:",printlist(clust,self.noun_cluster_size,True))
        #if(not self.noun in clust):
        #    clust.append(self.noun)
        return {
            'abstract' : clust[:self.noun_cluster_size],
            'concrete' : clust[-1*self.noun_cluster_size:]
        } 
    
            
    
    
    #********************************
    #   predicate synonym methods
    #********************************
    def wordnet_closure(self,word,pos):
        syn = set()
        for s in wn.synsets(word,pos):
            syn = syn.union(set([l.name() for l in s.lemmas() if l.name() != self.pred]))
        return list(syn)
    

    #*********************************
    #   candidate fetch methods
    #*********************************
    def combine_all(self,num):
        if num == 0:
            num = self.number_of_candidates
        def generatelist():
            print("fetching candidate replacements for "+self.pred)
            cands = set()
            candidates = eval(self.classname).pred_candidates
            added = cur =  0
            if not candidates.has(self.lnoun):
                print("can't find typical preds for ",self.noun)
                return None
            cands_max = candidates.get(self.lnoun)
            added = cur = 0
            while added < num and cur < len(cands_max):
                while cands_max[cur] == self.pred:
                    cur += 1
                candidate = cands_max[cur]
                if not candidate in vecs:
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
            wn_syns = self.wordnet_closure(self.pred,self.wordnet_class)
            # put the synonyms first while eliminating duplicates
            candlist = list(wn_syns) + [cand for cand in cands if cand not in wn_syns]
            if SingleWordData.empty(candlist):
                print("all ",len(cands_max)," candidates are either vectorless or too far from ",self.pred)
                return None
            print("found:",printlist(candlist,num,True))
            return candlist
        return generatelist
    
    

class AdjSubstitute(MetaphorSubstitute):
    object_clusters = AdjObjects(ngrams)
    pred_candidates = ObjectAdjs(ngrams) 
    #store the (filtered) synonym lists and the concrete/abstract
    #mean vectors corresponding to each adjective
    adjdata = dict()
    
    # it's not clear what Neuman et al mean by "most connected".
    # the graph is directed, so is it in, out or all archs of a node.
    # so I just compute all three.
    ccriterion = 'out'
    
    def __init__(self,options):
        self.wordnet_class = wn.ADJ
        self.strong_syns = list()
        super(AdjSubstitute,self).__init__(options)
        self.isnovel = self.isnovel()
            
        
        
    #******************************
    #   rating methods
    #******************************
    def pseudoneum(self):
        touchstone = clearvecs(self.orign_touchstone)
        def rate(cand):
            if cand in vecs:
                return vecs.n_similarity([cand],touchstone)
            return 0
        return rate

    def simpleadj(self,withnoun=False):
        def rate(cand):
            if cand in vecs:
                if self.lnoun in vecs and withnoun:
                    return vecs.n_similarity([cand],[self.pred,self.lnoun])
                return vecs.similarity(cand,self.pred)
            else:
                return 0
        return rate

    def adjabst(self,operation='mult',withnoun=False):
        def rate(cand):
            a = abst.get(cand) or 0.5
            if operation == 'mult':
                return a * self.simpleadj(withnoun)(cand)
            else:
                return a + self.simpleadj(withnoun)(cand)
        return rate
    
    def by_coca_synonyms_of_pred(self,num,withnoun=False):
        touchstone = clearvecs(self.coca_syns[:num])
        if withnoun and self.lnoun in vecs:
            touchstone.append(self.lnoun)
        def rate(cand):
            if cand not in vecs:
                return 0
            return vecs.n_similarity([cand],touchstone)
        return rate
    
    def coca_abstract(self,pool,num,withnoun=False):
        touchstone = sorted(clearvecs(self.coca_syns[:pool]),key=lambda x: abst.get(x),reverse=True)[:num]
        if withnoun and self.lnoun in vecs:
            touchstone.append(self.lnoun)
        def rate(cand): 
            if cand not in vecs:
                return 0
            return vecs.n_similarity([cand],touchstone)
        return rate
    
    def neuman_orig(self,include_noun=True):
        touchstone = clearvecs(self.neuman_filtered_synoyms())
        if include_noun and vecs.has(self.lnoun):
            print("including noun")
            touchstone.append(self.lnoun)
        def rate(cand):
            if vecs.has(cand):
                return vecs.n_similarity([cand],touchstone)
            else:
                print("not in vecs:",cand)
                return 0
        return rate
    
    def neuman_no_filtering(self,include_noun=True):
        touchstone = clearvecs(self.coca_syns)[:params['neuman_exp2']['thetanum']] + [self.pred]
        if include_noun and vecs.has(self.lnoun):
            touchstone.append(self.lnoun)
        def rate(cand):
            if vecs.has(cand):
                return vecs.n_similarity([cand],touchstone)
            return 0
        return rate

    def adjtimesfreq(self,num):
        def rate(cand):
            factor1 = self.coca_asbtract(10)(cand)
            factor2 = self.utsumi1cat(num)(cand)
            return factor1*factor2
        return rate
    
    def adjplusfreq(self,num):
        def rate(cand):
            factor1 = self.simpleadj(1)(cand)
            factor2 = self.utsumi1cat(num)(cand)
            return factor1 + factor2
        return rate
    
        
    def utsumi_comparison(self,evalmethod,basetype='simplified'):
        k = params['utsumi'][evalmethod]['comp']
        if basetype == 'original':
            base = self.utsumi_comparison_base1(k,300)
        elif basetype == 'simplified':
            base = self.utsumi_comparison_base2(k)
        def rate(cand):
            if cand in vecs:
                return vecs.n_similarity([cand],base)
            return 0
        return rate


    def utsumi1cat(self,evalmethod):
        m,k = params['utsumi'][evalmethod]['onecat']
        env = vecs.most_similar(self.pred,topn=m)
        T = self.utsumi_base(env)
        def rate(cand):
            if cand not in vecs:
                return 0
            return vecs.n_similarity([cand],[w[0] for w in T[:k]])
        return rate


    def utsumi2cat(self,evalmethod):
        m1,k,m2 = params['utsumi'][evalmethod]['twocat']
        env = vecs.most_similar(self.pred,topn=m1)
        T1 = self.utsumi_base(env)[:k]
        T1C = np.sum(vecs[w] for w,s in T1)/k
        T2 = [w[0] for w in vecs.similar_by_vector(T1C,topn=m2)]
        if self.lnoun in vecs:
            T2.append(self.lnoun)
        T2.append(self.pred)
        def rate(cand):
            if cand not in vecs:
                return 0
            return vecs.n_similarity([cand],T2)
        return rate

    #*******************************
    #   candidate fetch methods
    #*******************************
    def selfgold(self):
        def genlist():
            return self.gold
        return genlist

    def goldcands(self):
        def genlist():
            ret = set()
            for noun,dat in pairs[self.pred]['with'].items():
                ret = ret.union(set(dat['gold']))
            return list(ret)
        return genlist

    def semgold(self):
        def genlist():
            with open("semgold-candidates.txt") as sg:
                for l in sg.readlines():
                    if l.startswith(self.pred+".a"):
                        cands = l.split("::")[1].split(";")
                        cands[-1] = cands[-1].strip("\n")
                        return cands
            return []
        return genlist

    def coinco(self):
        def genlist():
            l = subprocess.check_output(['sh','coincoline.sh',self.pred],universal_newlines=True).split("\n\n") 
            ret = list()
            for w in l:
                if w != "" and w not in ret:
                    ret.append(w)
            print(ret)
            return ret
        return genlist
            
            
    def roget_abstract_top(self,num):
        def genlist():
            arog = sorted(self.roget_syns,key = lambda x: abst.get(x),reverse=True)
            return arog[:num]
        return genlist

    def all_dictionaries(self,wnspread):
        def genlist():
            w = set(self.wncands(wnspread)())
            c = set(self.coca_syns)
            r = set(self.roget_syns)
            return list(w.union(c).union(r))
        return genlist

    def all_dictionaries_abst(self,wnspread):
        def genlist():
            raw = self.all_dictionaries(wnspread)()
            return sorted(raw,key=lambda x: abst.get(x),reverse=True)[:15]
        return genlist

    def all_dicts_and_ngrams_filtered(self,wnspread,numgrams,radius):
        def genlist():
            touchstone = clearvecs(self.all_dictionaries_abst(1)())
            allsyns = self.all_dictionaries(wnspread)()
            mods = self.ngramcands(numgrams)()
            al = clearvecs(list(set(mods + allsyns)))
            #return [adj for adj in al if vecs.n_similarity(touchstone,[adj]) >= radius]
            #return squeeze(al,30)
            return self.neuman_filtered_synoyms(al)
        return genlist
    
    def all_dictionaries_squeezed(self,wnspread,size):
        def genlist():
            al = self.all_dictionaries(wnspread)()
            return squeeze(al,size)
        return genlist
    
    def synsormods(self,w,c,n):
        def genlist():
            if not self.isnovel:
                print("conventional")
                #self.candidate_rank = self.simpleadj(1)
                if len(applicable_synonyms) > 3: 
                    ret = list(applicable_synonyms)
                else:
                    ret = list(applicable_synonyms) #+ list(mods)[:(10 - len(applicable_synonyms))]
            else:
                print("novel")
                #self.candidate_rank = self.coca_abstract(12)
                ret =  list(mods)
            return ret
        return genlist
                

    def neumans_four(self):
        def genlist():
            return self.topfour
        return genlist
    
    def cocands(self,num): 
        def genlist():
            return self.coca_syns[:num]
        return genlist
    
    def wncands(self,spread):
        def genlist():
            l1 = self.wordnet_closure(self.pred,self.wordnet_class)
            if spread == 1:
                return l1
            else:
                l2 = set()
                for w in l1:
                    l2 = l2.union(set(self.wordnet_closure(w,self.wordnet_class)))
                    l2.add(w)
                return list(l2)
        return genlist
        
    def ngramcands(self,num):
        def genlist():
            bynoun = AdjSubstitute.pred_candidates.get(self.lnoun)
            if bynoun is not None:
                return [w[0] for w in  bynoun.most_common(num) if w[0] != self.pred]
            else:
                return []
        return genlist 
    
    def all_dicts_and_coca(self,w,n):
        def genlist():
            fromngrams = self.ngramcands(n)()
            fromdicts = self.all_dictionaries(w)()
            return list(set(fromngrams+fromdicts))
        return genlist
    
    def ngrams_wncands_cocands(self,n,w,c):
        def genlist():
            fromngrams = self.ngramcands(n)()
            fromwordnet = self.wncands(w)()
            fromcoca = self.cocands(c)()
            return list(set(fromcoca+fromngrams+fromwordnet))
        return genlist
    
    def rand3(self,num):
        def genlist():
            raw = AdjSubstitute.pred_candidates.get(self.lnoun).most_common(num)
            adjs = clearvecs(raw,0)
            random.shuffle(adjs)
            cands =  adjs[:3] + [self.neuman_correct]
            return cands
        return genlist
    
    def rand3_synonyms(self,wnspread):
        def genlist():
            syns = clearvecs(self.all_dictionaries(wnspread)())
            random.shuffle(syns)
            cands = syns[:3] + [self.neuman_correct]
            return cands
        return genlist

    def combine_all(self):
        def genlist():
            print("fetching candidate replacements for "+self.pred)
            cdatasource = eval(self.classname).pred_candidates
            if not cdatasource.has(self.lnoun):
                print("can't find typical preds for ",self.noun)
                return None
            cands_max = cdatasource.get(self.lnoun).most_common()
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
        return genlist

    #   helpers
    #-----------------------
    def utsumi_base(self,env):
        touchstone = env
        if self.lnoun in vecs:
            touchstone = sorted(env,key=lambda x: vecs.similarity(x[0],self.lnoun),reverse=True)
        else:
            print("instance noun not in vecs, using",len(env),"closest to",self.pred)
        return touchstone


    def utsumi_comparison_base1(self,k,samp):
        if self.lnoun not in vecs:
            return [w for w,s in vecs.most_similar(self.pred,topn=k)]
        base = list()
        adjenv = [w[0] for w in vecs.most_similar(self.pred,topn=samp)]
        nounenv = [w[0] for w in vecs.most_similar(self.lnoun,topn=samp)]
        for i in range(samp):
            inter = set([w for w in nounenv[:i]]).intersection([w for w in adjenv[:i]])
            if len(inter) >= k:
                return list(inter)
        print("entering recursion",2*samp)
        return self.utsumi_comparison_base1(k,2*samp)
    
    def utsumi_comparison_base2(self,k):
        if self.lnoun not in vecs:
            return [w[0] for w in vecs.most_similar(self.pred,topn=k)]
        instance_c = (vecs[self.lnoun] + vecs[self.pred])/2
        return [w for w,s in vecs.similar_by_vector(instance_c,topn=k)]

    def modifies_noun1(self,adj):
        objs = AdjSubstitute.object_clusters.get(adj)
        if self.lnoun in objs:
            return objs.freq(self.lnoun)
        return 0
    
    def modifies_noun2(self,adj):
        adjs = AdjSubstitute.pred_candidates.get(self.lnoun)
        if adj in adjs:
            return adjs.freq(adj)/adjs.N()
        if adjs.N() > 0:
            return 1/adjs.N()
        return 0
    
    def get_graph_data(self,kind):
        """
        reads the graph previously computed according to Neuman et al description.  
        
        :param kind: concrete/abstract 
        :return: list of prototypical abstract/concrete nouns modified by adj
        """
        #crit = NeumanAsIs.ccriterion
        crit = 'all'
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,self.pred+"-"+kind+".pkl"),'rb'))
        proto_nouns = [n[0] for n in data[crit].most_common() if n[0] in vecs][:NeumanGraph.most_connected]
        return proto_nouns
    

    def filter_antonyms(self,words):
        pass;
        #print("filtering antonyms")
        #rem = list()
        #for word in words:
        #    synsts = wn.synsets(word,self.wordnet_class)
        #    syns = clearvecs(self.wordnet_closure(word,None))
        #    for snset in synsts:
        #        for lem in snset.lemmas():
        #            ants = lem.antonyms()
        #            for ant in [a.name() for a in ants if a.name() in vecs]:
        #                remove  = ""
        #                add = ""
        #                print("checking",ant,"against",word)
        #                dis1 = vecs.n_similarity([word],syns)
        #                dis2 = vecs.n_similarity([ant],syns)
        #                if dis1 > dis2:
        #                    remove = ant
        #                elif dis2 > dis1:
        #                    remove = word
        #                    add = ant
        #                if add in vecs and add not in words:
        #                    print("adding",add)
        #                    #words.append(add)
        #                if remove not in rem and remove in words:
        #                    print("removing",remove)
        #                    words.remove(remove)
        #                    rem.append(remove)
        #return words

    #   miscelenia
    #------------------------
    def pred_antonyms(self):
        base = antonyms(self.pred)
        for ant in base:
            base = base.union(self.wordnet_closure(ant,self.wordnet_class))
        return base

    def neuman_filtered_synoyms(self,syns=None):
        if self.pred not in AdjSubstitute.adjdata:
            AdjSubstitute.adjdata[self.pred] = {
                'concrete_nouns' : self.get_graph_data('concrete'),
                'abstract_nouns' : self.get_graph_data('abstract'),
            }

        if 'filtered_syns' in AdjSubstitute.adjdata[self.pred]:
            return copy.copy(AdjSubstitute.adjdata[self.pred]['filtered_syns'])
        ret = list()
        simcut =  params['neuman_exp2']['thetacut']
        n = params['neuman_exp2']['thetanum']
        if syns is None:
            syns = clearvecs(self.coca_syns)[:n]
        for s in syns:
            sim_abst = vecs.n_similarity(AdjSubstitute.adjdata[self.pred]['abstract_nouns'], [s])
            sim_conc = vecs.n_similarity(AdjSubstitute.adjdata[self.pred]['concrete_nouns'],[s])
            print(s,sim_abst,sim_conc)
            if sim_abst > simcut and sim_conc < simcut:
                ret.append(s)
            else:
                print(s,"ruled out by protolist")
        print("in total ruled out for", self.pred+":",len(syns) - len(ret))
        # this was not clear in the paper, I'm assuming it makes sense
        ret.append(self.pred)
        AdjSubstitute.adjdata[self.pred]['syns'] = ret
        return ret

    def detect(self,adj):
        cparams = params['classifier']
        print("checking",adj,"literality")
        allnouns = eval(self.classname).object_clusters.get(adj).most_common(cparams['total'])
        concrete = [n[0] for n in sorted(allnouns,key=lambda x:
        abst.get(x[0]) if not SingleWordData.empty(x[0]) else 1)][:cparams['kappa']]
        print("concrete:",printlist(concrete,5,True))
        categories = self.concrete_categories(concrete)
        for cat in categories:
            if self.lnoun in nouncats[cat]:
                print(self.lnoun,"is listed under",cat+".",adj,"--",self.lnoun,"is literal.")
                return False #literal
        return True #metaphoric
        
    def concrete_categories(self,conclist):
        concrete_cats = list()
        for cat in nouncats:
            intersect = len(set(conclist).intersection(set(nouncats[cat])))
            if intersect >= params['classifier']['cat_thresh']:
                concrete_cats.append(cat)
        if len(concrete_cats) > 0:
            print("found categories:",printlist(concrete_cats,len(concrete_cats),True))
        else:
            print("no category found")
        return concrete_cats

class Irst2(AdjSubstitute):
    def __init__(self,options):
        self.cand_scores = list()
        self.target_grams = list()
        self.pred = options['pred']
        self.noun = options['noun']
        self.semid = options['semid']
        self.context = self.get_context() 
        super(Irst2,self).__init__(options)
    
    def resolve_methods(self):
        self.get_candidates = eval('self.'+self.methods['candidates'])
    
    def get_scores(self):
        scores = dict()
        cands = self.get_candidates()
        self.coverage = overlap(self.gold,cands)
        for place,cand in enumerate(cands):
            s = self.swaped_ngrams(cand)
            for i in range(2,6):
                if place == 0:
                    scores[i] = list()
                score = 0
                grams = [g for g in s if len(g) == i]
                for gram in grams:
                    score += ggrams.get(" ".join(gram))
                    #time.sleep(10)
                if score > 0:
                    scores[i].append((cand,score))
        ggrams.save_table()
        self.cand_scores = scores
        return scores

    def self_replaceability(self):
        grams = self.target_ngrams()
        counts = list()
        ans = 0
        for g in grams:
            count = ggrams.get(" ".join(g))
            counts.append((len(g),count))
        for i in range(2,6):
            iscore = sum([x[1] for x in counts if x[0] == i])
            ans += i*iscore
        return ans

    def target_ngrams(self):
        if not SingleWordData.empty(self.target_grams):
            return self.target_grams
        c = self.context
        tar = self.pred
        if tar not in c:
            print("cant find instance predicate in context")
        ans = list()
        for length in range(2,6):
            grams = nltk.ngrams(c,length)
            ans += [list(gram) for gram in grams if tar in gram]
        self.target_grams = ans
        return ans

    def swaped_ngrams(self,cand):
        grams = copy.deepcopy(self.target_ngrams())
        ans = list()
        for g in grams:
            tind = g.index(self.pred)
            g[tind] = cand
            if tind > 0 and (g[tind  - 1] == "an" or g[tind - 1] == "a"):
                g[tind - 1] = checkana(ggrams,cand)
            ans.append(g)
        del grams
        return ans
    
    def get_semeval_context(self):
        semid = str(self.semid)
        cont = semevaltest.xpath('//instance[@id="'+semid+'"]/context/text()')
        if len(cont) == 2:
            before,after = cont       
        else:
            c = cont[0]
            after = c if c.strip().startswith(self.pred) else ""
            before = c if after == "" else ""
        return before+" "+self.pred+" "+after

    def get_context(self):
        if params['pairs_file'] == 'sempairs.yml':
            cont = self.get_semeval_context()
        else:
            cont = pairs[self.pred]['with'][self.noun]['context']
        #return list(filter(lambda w: w not in string.punctuation,nltk.tokenize.word_tokenize(cont)))
        return nltk.tokenize.word_tokenize(AdjSubstitute.erlangen_swaps(cont))

    
    def find_substitutes(self):
        scores = self.get_scores()
        subs = list()
        done = set()
        keys = list(scores.keys())
        keys.reverse()
        for k in keys:
            for cand,score in sorted([p for p in scores[k]],key=lambda x: x[1],reverse=True):
                if cand not in done:
                    subs.append((cand,score))
                    done.add(cand)
        self.substitutes = subs
        return subs

class Irst22(Irst2):
    stashfile = os.path.join(SingleWordData.dbcache,'erlangenstash.pkl')
    def __init__(self,options):
        super(Irst22,self).__init__(options)
        if os.path.isfile(Irst22.stashfile):
            self.stash = pickle.load(open(Irst22.stashfile,'rb'))
        else:
            self.stash = dict()
        self.measure = 'frequency'
    
    def target_ngrams(self):
        grams = super(Irst22,self).target_ngrams()
        ret = list()
        for g in grams:
            gbar = copy.deepcopy(g)
            for i,w in enumerate(g):
                if g[i] == self.pred:
                    if g[i-1] in ('a','an','A','An'):
                        gbarbar = copy.deepcopy(g)
                        gbarbar[i-1] = g[i-1].strip('n') if 'n' in g[i-1] else gbar[i-1]+'n'
                        gbarbar[i] = "*"
                        ret.append(gbarbar)
                    gbar[i] = "*"
                    ret.append(gbar)
        return ret    
    
    def get_scores(self):
        scores = dict()
        cands = self.get_candidates()
        self.coverage = overlap(self.gold,cands)
        tgrams = self.target_ngrams()
        for place,cand in enumerate(cands):
            if cand in self.stash and self.semid in self.stash[cand]:
                print(cand,"scores from stash")
                for i in range(2,6):
                    if place == 0:
                        scores[i] = list()
                    s = self.stash[cand][self.semid][self.measure][i]
                    if s > 0:
                        scores[i].append((cand,s))
            else:
                candscores = np.zeros(6)
                for i in range(2,6):
                    if place == 0:
                        scores[i] = list()
                    score = 0
                    grams = [g for g in tgrams if len(g) == i]
                    for gram in grams:
                        d = ggrams2.get(gram)
                        if isinstance(d,pd.core.fram.DataFrame):
                            if cand not in d['N-gram'].values:
                            #print(cand,"not found",gram) 
                                continue 
                            else:
                                s = d[d['N-gram'] == cand]['frequency'].values[0]
                                score += s

                    if score > 0:
                        candscores[i] = score 
                        scores[i].append((cand,score))
                if cand not in self.stash and self.semid is not None:
                    self.stash[cand] = {
                        self.semid : {
                            self.measure: candscores
                        }
                    }
        self.cand_scores = scores
        with open(Irst22.stashfile,'wb') as f:
            pickle.dump(self.stash,f)
        return scores
    
    def replaceability(self,word=None):
        if word is None:
            word = self.pred
        ans = 0
        for g in self.target_ngrams():
            d = ggrams2.get(g)
            if isinstance(d,pd.core.frame.DataFrame) and  word in d['N-gram'].values:
                f = d[d['N-gram'] ==  word]['frequency'].values[0]
                p = percentileofscore(d['frequency'],f)/100
                ans += len(g)*p
        return ans/(sum(range(2,6)))
    
        

    def self_rrep(self):
        return self.replaceability()

class Irst3(Irst22):
    def find_substitutes(self):
        return super(Irst2,self).find_substitutes()
    
    def resolve_methods(self):
        super(Irst2,self).resolve_methods()
    
    # candidate rating #
    #------------------#
    def rep(self):
        scores = self.get_scores()
        #sums = dict((i,sum([t[1] for t in scores[i]])) for i in scores.keys())
        sums = dict()
        for k in scores.keys():
            sums[k] = nltk.MLEProbDist(nltk.FreqDist(dict(scores[k])))
        def rate(cand):
            r = 0
            for ngramlength,dat in scores.items():
                for c,s  in dat:
                    if c == cand:
                        r += ngramlength*sums[ngramlength].prob(c)
            return r
        return rate
    
    def simplerep2(self):
        def rate(cand):
            return self.replaceability(cand)
        return rate

    def simplerep(self):
        scores = self.get_scores()
        def rate(cand):
            ans = 0
            for i in range(2,6):
                iscore = sum([x[1] for x in scores[i] if x[0] == cand])
                ans += i*iscore
            return ans
        return rate
    
    def self_rrep(self):
        r = super(Irst3,self).self_replaceability()
        rep = self.simplerep()
        rs = [rep(c) for c in self.get_candidates()]
        return percentileofscore(rs,r)/100

    def repsim(self):
        r = self.rep()
        sim = self.coca_sbstract(30,12)
        def rate(cand):
            return r(cand) * sim(cand)
        return rate
    
    def origrepsim(self):
        subs = [w for w,r in super(Irst3,self).find_substitutes()]
        #sim = self.coca_abstract(30,12)
        #sim = self.by_coca_synonyms_of_pred(12)
        sim = self.adjabst()
        sr = self.self_rrep()
        def rate(cand):
            if cand not in subs:
                return sim(cand)
            return sim(cand) + 1/(subs.index(cand)+1)
        return rate

class SemEvalSystem(AdjSubstitute):
    datadir = "../semeval/systems"
    def __init__(self,options):
        options['dry_run'] = True
        super(AdjSubstitute,self).__init__(options)
        self.oot_file = self.filepath('oot')
        self.best_file = self.filepath('best')
        self._mode = 'oot'
        self.resultsfile = self.oot_file
    
    @property
    def mode(self):
        return self._mode
        
    @mode.setter
    def mode(self,mode):
        self._mode = mode
        self.resultsfile = eval('self.'+mode+'_file')

    def filepath(self,kind):
        return os.path.join(SemEvalSystem.datadir,"{}.{}".format(self.system_name,kind))
    
    def hasresults(self):
        return os.path.isfile(eval('self.'+self._mode+'_file'))
    
    def find_substitutes(self):
        subs = list()
        results = self.parse_results_line() 
        if results is not None:
            for i,r in enumerate(results,1):
                subs.append((r.strip("\n"),1/i))
            self.substitutes = subs
        return subs
    
    def parse_results_line(self):
        if not self.hasresults():
            return None
        # find the line by pred and noun
        for l in open(self.resultsfile).readlines():
            components = l.split(" ")
            if components[0] == self.pred+".a" and int(components[1]) == int(self.semid):
                return components[3].split(";")
            
        
