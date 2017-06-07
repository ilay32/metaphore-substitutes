#   imports
#--------------------
import random,copy,re,nltk,string,time
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from dbutils import *
from prygress import progress
from ngraph import NeumanGraph
from sklearn.cluster import KMeans as km
from scipy.stats import spearmanr
from lxml import etree


#   globals
#----------------------
ggrams = Erlangen()
params = yaml.load(open('params.yml'))
pairs = yaml.load(open(params['pairs_file']))
nouncats = pickle.load(open('nclass.pkl','rb'))
ngrams = DB('NgramsCOCAAS')
abst = Abst(ngrams)
lemmatizer = WordNetLemmatizer()
vectormodel = conf['vectormodel']
semevaltest = etree.parse('../semeval/lexsub_test.xml')
if vectormodel == "LSA":
    vecs = Vecs(DB('LSA-ENG'))
elif vectormodel == "SPVecs":
    vecs = SPVecs()
else:
    vecs = Word2Vec.load_word2vec_format(vectormodel,binary=True) 
    vecs.has = lambda x: x in vecs
    vecs.get = lambda x: vecs[x]
    vecs.centroid = lambda l: vecs[vecs.most_similar(l,topn=1)[0][0]]



#   helpers
#-----------------
def clearvecs(l,ind=None):
    ret = list()
    for x in l:
        y = x[ind] if ind is not None else x
        if y in vecs:
            ret.append(y)
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
        self.noun = lemmatizer.lemmatize(self.noun)
        self.substitutes = list()
        self.no_abst = set()
        self.no_vector = set()
        self.too_far = set()
        self.classname = self.__class__.__name__
        if not self.dry_run:
            self.resolve_methods()    
        if params['pairs_file'] == 'sempairs.yml':
            self.mode = self.gold[0]
    
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

        *get_candidates* and *rank candidates*
        are specified as parameters to be sorted out by *resolve_methods*.
        this allows for maximal flexibility without a deep and branchy
        inheritance tree.
        """
        candidates = self.get_candidates()
        subs = list()
        if not SingleWordData.empty(candidates):
            for pred in candidates:
                if pred in vecs:
                    subs.append((pred,self.candidate_rank(pred)))
                else:
                    subs.append((pred,-1))
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
    
    #******************************* 
    #   evaluation methods           
    #*******************************
    @preeval
    def oot(self):
        gld = self.gold
        return sum([1 for s in self.substitutes[:10] if s in gld])/len(gld)

    @preeval
    def best(self):
        gld = self.gold
        return sum([1 for s in self.substitutes if s in gld])/(len(gld)*len(self.substitutes))


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
    def spear(self):
        ranks = list()
        gld = self.gold
        sampsize = min(len(gld),len(self.substitutes))
        for sub in self.substitutes[:sampsize]:
            if sub[0] in gld:
                ranks.append(gld.index(sub[0]))
        if len(ranks) > 1:
            s = spearmanr(ranks,sorted(ranks))
            return s.correlation
        # a little patch for the cases where only one of the substitutes
        # is a memeber of the gold set.
        if len(ranks) == 1:
            return 1 - ranks[0]/len(gld)
        return 0
    
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
        return int(self.substitute() == self.mode)


    #********************** 
    #   rating methods      
    #**********************
      

    #   miscelenia
    #----------------------
        
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
            if not candidates.has(self.noun):
                print("can't find typical preds for ",self.noun)
                return None
            cands_max = candidates.get(self.noun)
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
            
        
        
    #******************************
    #   rating methods
    #******************************
    def simpleadj(self,num):
        def rate(cand):
            if cand in vecs:
                return vecs.similarity(cand,self.pred)
            else:
                return 0
        return rate

    def by_coca_synonyms_of_pred(self,num):
        touchstone = clearvecs(self.coca_syns[:num],0)
        def rate(cand):
            return vecs.n_similarity([cand],touchstone)
        return rate
    
    def coca_with_noun(self,num):
        touchstone =  clearvecs(self.coca_syns[:num],0) + [self.noun]
        def rate(cand):
            return vecs.n_similarity([cand],touchstone)
        return rate
    
    def coca_abstract(self,num):
        touchstone = sorted(clearvecs(self.coca_syns),key=lambda x: abst.get(x),reverse=True)[:num]
        def rate(cand): 
            if cand not in vecs:
                return 0
            return vecs.n_similarity([cand],touchstone)
        return rate
    
    def neuman_orig(self,num):
        touchstone = neuman_filtered_synoyms(self.pred)[:num] + [self.noun]
        def rate(cand):
            return vecs.n_similarity([cand],touchstone)
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
    
    def utsumi1cat(self,num):
        def rate(cand):
            if cand not in vecs:
                return 0
            env = vecs.most_similar(self.pred,topn=50)
            env.sort(key=lambda x: vecs.similarity(x[0],self.noun),reverse=True)
            return vecs.n_similarity([cand],[w[0] for w in env[:num]])
        return rate



    #*******************************
    #   candidate fetch methods
    #*******************************
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

    def all_dicts_and_ngrams_filtered(self,wnspread,ngrams,radius):
        def genlist():
            touchstone = clearvecs(self.all_dictionaries_abst(1)())
            allsyns = self.all_dictionaries(wnspread)()
            mods = self.ngramcands(ngrams)()
            al = clearvecs(list(set(mods + allsyns)))
            #return [adj for adj in al if vecs.n_similarity(touchstone,[adj]) >= radius]
            return squeeze(al,30)
        return genlist
    
    def all_dictionaries_squeezed(self,wnspread,size):
        def genlist():
            al = self.all_dictionaries(wnspread)()
            return squeeze(al,size)
        return genlist
    
    def synsormods(self,w,c,n,r2):
        def genlist():
            wnsyns = self.wncands(w)()
            cocsyns = self.cocands(c)()
            syns = set(wnsyns + cocsyns)
            mods = set(self.ngramcands(n)())
            applicable_synonyms = syns.intersection(mods)
            ret = None
            if len(applicable_synonyms) > 0:
                self.candidate_rank = self.simpleadj(1)
                if len(applicable_synonyms) > 3: 
                    ret = list(applicable_synonyms)
                else:
                    ret = list(applicable_synonyms) #+ list(mods)[:(10 - len(applicable_synonyms))]
            else:
                self.candidate_rank = self.adjplusfreq(15)
                ret =  list(mods)
            return ret
        return genlist
                

    def neumans_four(self,num):
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
            bynoun = AdjSubstitute.pred_candidates.get(self.noun).most_common(num)
            if bynoun is not None:
                return [w[0] for w in bynoun if w[0] != self.pred]
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
            raw = AdjSubstitute.pred_candidates.get(self.noun).most_common(num)
            adjs = clearvecs(raw,0)
            random.shuffle(adjs)
            cands =  adjs[:3] + [self.mode]
            return cands
        return genlist
    
    def rand3_synonyms(self,wnspread):
        def genlist():
            syns = clearvecs(self.all_dictionaries(wnspread)())
            random.shuffle(syns)
            cands = syns[:3] + [self.mode]
            return cands
        return genlist

    def combine_all(self):
        def genlist():
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
        return genlist

    #   helpers
    #-----------------------
    
    def modifies_noun1(self,adj):
        objs = AdjSubstitute.object_clusters.get(adj)
        if self.noun in objs:
            return objs.freq(self.noun)
        return 0
    
    def modifies_noun2(self,adj):
        adjs = AdjSubstitute.pred_candidates.get(self.noun)
        if adj in adjs:
            return adjs.freq(adj)/adjs.N()
        return 1/adjs.N()
    
    def get_graph_data(self,kind):
        """
        reads the graph previously computed according to Neuman et al description.  
        
        :param kind: concrete/abstract 
        :return: list of prototypical abstract/concrete nouns modified by adj
        """
        crit = NeumanAsIs.ccriterion
        data = pickle.load(open(os.path.join(NeumanGraph.datadir,self.pred+"-"+kind+".pkl"),'rb'))
        proto_nouns = [n[0] for n in data[crit].most_common() if n[0] in vecs][:NeumanGraph.most_connected]
        return proto_nouns
    

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

    #   miscelenia
    #------------------------
    def neuman_filtered_synoyms(self):
        if self.pred not in AdjSubstitute.adjdata:
            AdjSubstitute.adjdata[self.pred] = {
                'concrete_nouns' : NeumanAsIs.get_graph_data(self.pred,'concrete'),
                'abstract_nouns' : NeumanAsIs.get_graph_data(self.pred,'abstract'),
            }

        if 'filtered_syns' in NeumanAsIs.adjdata[self.pred]:
            return copy.copy(NeumanAsIs.adjdata[self.pred]['filtered_syns'])
        ret = list()
        simcut =  self.params['thetacut']
        n = self.params['thetanum']
        syns = [a for a in pairs[self.pred]['coca_syns'] if a in vecs][:n]
        for s in syns:
            sim_abst = vecs.n_similarity(NeumanAsIs.adjdata[self.pred]['abstract_nouns'], [s])
            sim_conc = vecs.n_similarity(NeumanAsIs.adjdata[self.pred]['concrete_nouns'],[s])
            if sim_abst > simcut and sim_conc < simcut:
                ret.append(s)
            else:
                print(s,"ruled out by protolist")
        print("in total ruled out for", self.pred+":",len(syns) - len(ret))
        # this was not clear in the paper, I'm assuming it makes sense
        ret.append(self.pred)
        NeumanAsIs.adjdata[self.pred]['syns'] = ret
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
            if self.noun in nouncats[cat]:
                print(self.noun,"is listed under",cat+".",adj,"--",self.noun,"is literal.")
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
        super(Irst2,self).__init__(options)
        self.cand_scores = list()
        self.target_grams = list()
        self.context = self.get_context() 

    def get_scores(self):
        scores = dict()
        for place,cand in enumerate(self.get_candidates()):
            s = self.swaped_ngrams(cand)
            for i in range(2,6):
                if place == 0:
                    scores[i] = list()
                score = 0
                grams = [g for g in s if len(g) == i]
                for gram in grams:
                    score += ggrams.get(" ".join(gram))
                    time.sleep(10)
                scores[i].append((cand,score,i))
        self.cand_scores = scores
        return scores

    
    def target_ngrams(self):
        if not SingleWordData.empty(self.target_grams):
            return self.target_grams
        c = self.context
        tar = self.pred
        if tar not in c:
            print("cant find target ngrams in contex")
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
        return list(filter(lambda w: w not in string.punctuation,nltk.tokenize.word_tokenize(cont)))
    
    def find_substitutes(self):
        scores = self.get_scores()
        subs = list()
        keys = list(scores.keys())
        keys.reverse()
        for k in keys:
            subs +=  sorted([trip for trip in scores[k] if trip[1] > 0],key=lambda x: x[1],reverse=True)
        self.substitutes = subs
        return subs
