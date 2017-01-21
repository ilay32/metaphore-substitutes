import pandas,sys,os,yaml,pymssql,math,copy,pickle,json,numbers,six,io,time,threading,re,subprocess,nltk
import numpy as np
import _thread as thread
from scipy.spatial.distance import cosine
from datetime import datetime
conf = yaml.load(open('params.yml'))

def hereiam(fn):
    def inside(*args,**kwargs):
        print("entering", fn.__name__)
        ans = fn(*args,**kwargs)
        print("exiting",fn.__name__)
        return ans
    return inside

def dberror(ret=None):
    def wrap(fn): 
        def inside(*args,**kwargs):
            try:
                ans = fn(*args,**kwargs)
            except Exception as e:
                print(e)
                ans = ret
            return ans
        return inside
    return wrap


class DB:
    def __init__(self,catalog=None):
        specs = yaml.load(open('db.yml'))
        self.server_name = specs['server']
        self.user = specs['user']
        self.password = specs['pass']
        self.catalog = catalog
        self.abst = dict()
        self.vecs = dict()
        self.conn = self.connect()
        self.connected = isinstance(self.conn,pymssql.Connection)
    
    def db_available(self):
        rcode = subprocess.run(["ping","-c 1",self.server_name],stdout = subprocess.PIPE)
        return rcode.returncode == 0

    def connect(self):
        cu = None
        if self.db_available():
            try:
                cu =  pymssql.connect(self.server_name, self.user, self.password,self.catalog)
            except:
                print("could not connect to",self.catalog,"working offline")
                cu = None
        return cu

    def query(self,q,tries=1):
        if not self.connected:
            return
        if tries > 1:
            conn = self.connect()
        else:
            conn = self.conn
        c = conn.cursor()
        c.setoutputsize(conf['cluster_maxrows'] + conf['candidates_maxrows'])
        if tries < 6:
            timer = threading.Timer(float(tries*3),self.query,[q,tries+1])
            if(tries > 1):
                print(q,"try",tries,"of 5")
            timer.start()
            try:
                #timer.join(float(tries*3 + 1))
                c.execute(q)
                if c.rownumber == 0:
                    timer.cancel()
            except pymssql.DatabaseError as e: 
                print("trying to execute ",q," got ",e)
                timer.join()
                timer.cancel()
            finally:
                timer.cancel()
                if conn != self.conn:
                    conn.close()
        else:
            print("tried",q,tries - 1,"times. giving up")
        return c  
        
    def destroy(self):
        if self.connected:
            self.conn.close()

class SingleWordData:
    dbcache = 'newdbcache'
    def __init__(self,db):
        self.db = db
        self.table_path = self.get_table_path()
        self.empty_table = dict()
        self.table = self.get_table()
        self.table_changed = False
        self.notfound = set()
    
    def searchval(self,val):
        for w,v in self.table.items():
            if v == val:
                return w
    def has(self,word):
        word = word.replace("'","")
        if word not in self.notfound:
            self.get(word)
            if word in self.table:
                return True
            self.notfound.add(word)
        return False
    
    def empty(obj):
        if not obj:
            return True
        if isinstance(obj,int) or  isinstance(obj,float):
            return obj == 0
        if isinstance(obj,str):
            return obj.isspace() or obj == ""
        notempty = False
        try:
            iterator = iter(obj)
            for x in iterator:
                notempty = not SingleWordData.empty(x)
        except:
            print("can't determine emptiness of type",type(obj))
            return False
        return len(obj) ==  0 or not notempty
    
    def get(self,word):
        if SingleWordData.empty(word):
            return None
        word = word.replace("'","")
        if word in self.notfound:
            print("this shouldn't be here:",word)
        if word in self.table:
            return self.table[word]
        elif self.db.connected:
            cu  = self.db.query(self.queryscheme(word))
            qr = self.handlequery(cu)
            if not SingleWordData.empty(qr):
                self.table[word] = qr
                self.table_changed = True
            return qr
        else:
            return None
    
    def __str__(self):
        return self.table[:min(len(self.table),5)]
    
    def get_table_path(self):
        return os.path.join(SingleWordData.dbcache,self.search_table()+'.pkl')
    
    def get_table(self):
        if os.path.isfile(self.table_path): 
            table = pandas.read_pickle(self.table_path)
        else:
            table = self.empty_table
        return table
    
    def save_table(self):
        if self.table_changed:
            print("saving",self.table_path)
            with open(self.table_path, 'wb') as f:
                pickle.dump(self.table,f)
    
    def destroy(self):
        self.save_table()
        self.db.destroy()


class Abst(SingleWordData):
    def queryscheme(self,word):
        return "SELECT ABSTRACT_SCALE FROM PHRASE_ABSTRACT WHERE PHRASE='{0}'".format(word.lower())
    
    @dberror(0)
    def handlequery(self,cu):
        f = cu.fetchall()
        if SingleWordData.empty(f):
            return 0.5
        return f[0][0]
    
    def search_table(self):
        return 'abst'

class Vecs(SingleWordData):
    def __contains__(self,word):
        return self.has(word)

    def queryscheme(self,word):
        return "GetRows '{0}'".format(word.lower())
    
    def n_similarity(self,ws1,ws2):
        if SingleWordData.empty(ws1) or SingleWordData.empty(ws2):
            print("empty argument for n_similarity")
            return
        v1 = self.centroid(ws1) if len(ws1) > 1 else self.get(ws1[0])
        v2 = self.centroid(ws2) if len(ws2) > 1 else self.get(ws2[0])
        try: 
            ret = Vecs.vec_similarity(v1,v2)
        except Exception as e:
            print(str(e))
            return
        return ret
     
    def centroid(self,words):
        if SingleWordData.empty(words):
            print("empty argument to Vecs.centroid")
            return None
        cluster = [w for w in words if self.has(w)]
        if SingleWordData.empty(cluster):
            print("no vector for any of the words in this list")
            return None
        acc = self.get(cluster[0])
        added = 1
        for word in cluster[1:]:
            if SingleWordData.empty(self.get(word)): 
                print("empty vector", word)
            else:
                acc = Vecs.addition(acc,self.get(word))
                added += 1
        return Vecs.multiply(acc,1/added)


    @dberror()
    def handlequery(self,query):
        ret = dict()
        for row in query:
            ret.update({row[2] : row[3]})
        return ret
   
    def vec_similarity(u,v):
        norms = Vecs.norm(u) * Vecs.norm(v)
        if norms == 0:
            return -1
        return Vecs.dot(u,v)/norms
    
    def dot(u,v):
        d = 0
        for colid,uval,vval in Vecs.vectorpair(u,v):
            d += uval*vval
        if d == 0:
            print("zero norm")
        return d
    
    def similarity(self,w1,w2):
        if self.has(w1) and self.has(w2):
            return Vecs.vec_similarity(self.get(w1),self.get(w2))
        return float("nan")

    def vectorpair(u,v):
        us = set(u.keys())
        vs = set(v.keys())
        ku = list(us.union(vs))
        for col_id in ku:
            vvalue = v[col_id] if col_id in v else 0
            uvalue = u[col_id] if col_id in u else 0
            yield(col_id,vvalue,uvalue)
    
    def search_table(self):
        return 'vecs'
    # u and v are not of equal length, so this norm is conditioned on shared coordinates (just like the inner product) 
    def norm(u):
        return math.sqrt(Vecs.dot(u,u))

    def addition(u,v):
        s = dict()
        for colid,uval,vval in Vecs.vectorpair(u,v):
            s.update({colid:uval+vval})
        return s
    
    def multiply(u,scalar):
        ret =  copy.deepcopy(u)
        for k in ret.keys():
            ret[k] = u[k]*scalar
        return ret
    
    def subtract(u,v):
        return Vecs.addition(u,Vecs.multiply(v,-1))

class Lex(SingleWordData):
    @dberror() 
    def handlequery(self,q):
        ret = tuple()
        for r in q:
            ret += (r[0],)
        return ret
    
    def queryscheme(self,word):
        return "SELECT DISTINCT posType FROM Lexicon WHERE word='{0}' AND lemma = '{0}'".format(word)
    
    def search_table(self):
        return 'lex'

class Ngram(SingleWordData):
    def __init__(self,db):
        self.empty_table = nltk.ConditionalFreqDist()
        self.lem_column = 10
        self.freq_column = 1
        self.global_limit = conf['cluster_maxrows']
        super(Ngram,self).__init__(db)

    def search_table(self):
        return '-'.join([
            self.name,
            self.left,
            self.right,
            self.mi
        ])
    
        
    def queryscheme(self,word):
        query_base = "'{0}',{1},{2},{3},{4},{5}".format(
            word,
            self.key_pos, # this is the POS type of verb
            self.search_pos,
            self.left, # how many words to look behind
            self.right, # how many words to look ahead
            self.mi # minimal mutual information required for results
        )
        if self.db.connected:
            self.db.conn.cursor().execute("IsNgramsReady "+query_base+",1")
        time.sleep(2)
        return "GetNgrams "+query_base    
    
    @dberror()
    def handlequery(self,c):
        freqs = dict()
        cur = 0
        rows = c.fetchall()
        while cur < min(self.global_limit,len(rows)):
            freqs[rows[cur][self.lem_column]] = rows[cur][self.freq_column] 
            cur += 1
        return nltk.FreqDist(freqs)

class VerbObjects(Ngram):
    def __init__(self,db):
        self.key_pos = 2
        self.search_pos = 1
        self.left = str(conf['search_objects_left'])
        self.right = str(conf['search_objects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.name = 'object-clusters'
        super(VerbObjects,self).__init__(db)

class AdjObjects(Ngram):
    def __init__(self,db):
        self.key_pos = 3
        self.search_pos = 1
        self.left = str(conf['search_aobjects_left'])
        self.right = str(conf['search_aobjects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.name = 'adj-object-clusters'
        super(AdjObjects,self).__init__(db)

class ObjectAdjs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 3
        self.left = str(conf['search_aobjects_right'])
        self.right = str(conf['search_aobjects_left'])
        self.mi = str(conf['verb_min_MI'])
        self.name = 'aobject-candidates'
        super(ObjectAdjs,self).__init__(db)

class VerbSubjects(Ngram):
    def __init__(self,db):
        self.key_pos = 2
        self.search_pos = 1
        self.left = str(conf['search_subjects_left'])
        self.right = str(conf['search_subjects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.name = 'subject-clusters'
        super(VerbSubjects,self).__init__(db)

class ObjectVerbs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 2
        self.left = str(conf['search_object_verbs_left'])
        self.right = str(conf['search_object_verbs_right'])
        self.mi = str(conf['verb_min_MI'])
        self.name = 'object-candidates'
        super(ObjectVerbs,self).__init__(db)

class SubjectVerbs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 2
        self.left = str(conf['search_subject_verbs_left'])
        self.right = str(conf['search_subject_verbs_right'])
        self.mi = str(conf['verb_min_MI'])
        self.name = 'subject-candidates'
        super(SubjectVerbs,self).__init__(db)

class NounNoun(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 1
        self.left = str(conf['neuman_graph']['nn_window'])
        self.right = self.left
        self.mi = str(conf['noun_min_MI'])
        self.name = 'nn-cluster'
        super(NounNoun,self).__init__(db)

#
# general utility functions
#
def printlist(l,k=5,silent=False,separator=","):
    i = 0
    last = min(k,len(l)) - 1
    out = io.StringIO()
    if isinstance(l,set):
        l = list(l)
    while i <= last:
        if isinstance(l[i],(numbers.Number,six.string_types,tuple)):
            out.write(str(l[i]))
        elif isinstance(l[i],list):
            out.write(printlist(min(1,k-1),len(l[i]),True))
        elif isinstance(l[i],dict):
            out.write(printdict(l[i]))
        if i == last and i < len(l) - 1:
            out.write("...("+str(len(l))+")")
        if i < last:
            out.write(separator)
        i += 1
    ret = out.getvalue()
    out.close()
    if not silent:
        print(ret)
    return ret

def printdict(d,silent=False):
    out = io.StringIO()
    for i,v in d.items():
        out.write(str(i))
        out.write(":\t")
        if isinstance(v,(numbers.Number,six.string_types,tuple)):
            out.write(str(v))
        elif isinstance(v,list):
            out.write(printlist(v,len(v),True))
        elif isinstance(v,dict):
            out.write(printdict(v,True))
        out.write("\n")
    ret = out.getvalue()
    out.close()
    if not silent:
        print(ret)
    else:
        return ret

# explore the db cache
def explore_cache():
    dbcache = SingleWordData.dbcache
    cached = os.listdir(dbcache)
    for i,cache in enumerate(cached):
        print(str(i+1)+".\t",os.path.basename(cache))
    cache = int(input("choose one of the above: "))
    path = os.path.join(dbcache,cached[cache-1])
    if os.path.isfile(path):
        data = pandas.read_pickle(path)
        #ret = printdict(data)
        print("the "+cached[cache-1]+" basic dict is available as 'data'")
        return data
    print("failed to load ",path)
    return False

class RunData:
    datadir = "rundata" 
    def __enter__(self):
        return self
    
    def __exit__(self,typer, value, traceback):
        print("done")    
    
    def __init__(self,df,note,typ):
        self.data = df
        self.params = yaml.load(open('params.yml').read())
        self.timestamp = int(time.time())
        self.note = note.strip()
        self.typ = typ
    
    def __str__(self):
        out = io.StringIO()
        out.write(time.ctime(self.timestamp))
        out.write(" "+self.typ+" "+self.note)
        out.write("\nNeuman Score: "+self.neuman_eval())
        out.write("\nMRR: "+self.evaluate())
        return out.getvalue()

    def __repr__(self):
        return "<{0}>".format(self.__class__)
    
    def tofile(self,filename=None):
        if not filename:
            filename  = self.note.replace(' ','_')
        with open(os.path.join(filename+".txt"),"w") as f:
            d = self.data
            f.write(str(self)+"\n")
            for i,row in d.iterrows():
                f.write(row.pred+" "+row.noun+" ("+self.typ+", expected "+row.correct+")")
                f.write("\n=====================\n")
                for sub in row['substitutes'][:self.params['number_of_candidates']]:
                    f.write(sub[0]+" "+str(round(sub[1],5))+"\n")
                f.write("\n\n")
            
            
    
    def save(self):
        with open(os.path.join(RunData.datadir,str(self.timestamp)+".pkl"),"wb") as f:
            pickle.dump(self,f)

    def when(self):
        print(time.ctime(self.timestamp))

    def how(self):
        printdict(self.params)
        print("note: ",self.note)
        print("class: ",self.typ)
    
    def howmany(self) :
        return len(self.data)

    def show(self,pred=None,noun=None):
        d = self.data
        if pred == 'pairs':
            for i,row in d.iterrows():
                print(str(i+1)+".",row.pred," ",row.noun,"\n")
            return

        if pred in ['no_vector','no_abst','ignored']:
            acc = set()
            for i,row in d.iterrows():
                acc = acc.union(row[pred])
            printlist(acc,len(acc),False,"\n")
            return
        
        pairs = list()
        if pred!=None and noun!=None:
            pairs = d[(d.pred == pred) & (d.noun == noun)]
        elif pred != None and noun == None:
            pairs = d[d.pred == pred]
        elif noun != None and pred == None:
            pairs =  d[d.noun == noun]
                    
        else:
            pairs = d
        for i,row in pairs.iterrows():
            print(row.pred," ",row.noun," (expected", row.correct,"):")
            printlist(row['substitutes'],len(row['substitutes']),False,"\n")
            print("\n")
    
    def evaluate(self):
        return str(self.data['score'].mean())

    def neuman_eval(self):
        frac = self.data['neuman_score'].sum()/self.howmany()
        return "{:.1%}".format(frac)

class SPVecs:
    def __init__(self):
        self.dim = 300
        with open('nspvecs.pkl','rb') as t:
            self.table = pickle.load(t)
    
    def __contains__(self,word):
        return word in self.table
    
    def n_similarity(self,ws1,ws2):
        if SingleWordData.empty(ws1) or SingleWordData.empty(ws2):
            print("empty argument for n_similarity")
            return
        v1 = self.centroid(ws1) if len(ws1) > 1 else self.get(ws1[0])
        v2 = self.centroid(ws2) if len(ws2) > 1 else self.get(ws2[0])
        if SPVecs.check_two(v1,v2):
            return SPVecs.vec_similarity(v1,v2)
        print("None in n_similarity")
        return
    
    def check_two(v1,v2):
        return isinstance(v1,np.ndarray) and isinstance(v2,np.ndarray)

    def similarity(self,w1,w2):
        v1 = self.get(w1)
        v2 = self.get(w2)
        if SPVecs.check_two(v1,v2):
            return SPVecs.vec_similarity(v1,v2)
        print("None in similarity")
        return None

    def vec_similarity(v1,v2):
        return v1.dot(v2)/np.linalg.norm(v1)*np.linalg.norm(v2)
    
    def get(self,word):
        if word in self.table:
            return self.table[word]
        return None

    def centroid(self,words):
        ret = np.zeros(self.dim)
        for w in words:
            if w in self.table:
                ret = np.add(ret,self.table[w])
            else:
                print(w,"not in sp vectors")
        return ret/len(words) 

    def doesnt_match(self,words):
        cent = self.centroid(words)
        o = [(word,SPVecs.vec_similarity(self.get(word),cent)) for word in words if word in self]
        o.sort(key=lambda x: x[1])
        return o[0][0]


if __name__ == '__main__':
    print("to explore the cache type data=explore_cache()")
