import pandas,sys,os,yaml,pymssql,math,copy,pickle,json,numbers,six,io,time,threading,re
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
        self.connected = True
        self.conn = self.connect()
    
    def connect(self):
        try:
            return pymssql.connect(self.server_name, self.user, self.password,self.catalog)
        except:
            self.connected = False
            return None

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
    dbcache = 'dbcache'
    def __init__(self,db):
        self.db = db
        self.table_path = self.get_table_path()
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
            return obj.isspace()
        return len(obj) ==  0
    
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
    
    def __str__(self):
        return self.table[:min(len(self.table),5)]
    
    def get_table_path(self):
        return os.path.join(SingleWordData.dbcache,self.search_table()+'.pkl')
    
    def get_table(self):
        if os.path.isfile(self.table_path): 
            table = pandas.read_pickle(self.table_path)
        else:
            table = dict()
        return table
    
    def save_table(self):
        if self.table_changed:
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
        if (not cu) or (len(f) == 0):
            return 0.5
        return f[0][0]
    
    def search_table(self):
        return 'abst'

class Vecs(SingleWordData):
    def queryscheme(self,word):
        return "GetRows '{0}'".format(word.lower())
    
    @dberror()
    def handlequery(self,query):
        ret = dict()
        for row in query:
            ret.update({row[2] : row[3]})
        return ret
   
    def distance(u,v):
        norms = Vecs.norm(u) * Vecs.norm(v)
        if norms == 0:
            return -1
        return Vecs.dot(u,v)/norms
       # m = min(len(u),len(v))
       # return cosine(u[:m],v[:m])
    def dot(u,v):
        d = 0
        for colid,uval,vval in Vecs.vectorpair(u,v):
            d += uval*vval
        if d == 0:
            print("zero norm")
        return d
    
    def word_distance(self,w1,w2):
        if self.has(w1) and self.has(w2):
            return Vecs.distance(self.get(w1),self.get(w2))
        return float("inf")

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
        ret = set()
        cur = 0
        rows = c.fetchall()
        while len(ret) < self.global_limit and cur < len(rows):
            ret.add(rows[cur][self.column])
            cur += 1
        return list(ret)

class VerbObjects(Ngram):
    def __init__(self,db):
        self.key_pos = 2
        self.search_pos = 1
        self.left = str(conf['search_objects_left'])
        self.right = str(conf['search_objects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.column = 10
        self.global_limit = conf['cluster_maxrows']
        self.name = 'object-clusters'
        super(Ngram,self).__init__(db)

class AdjObjects(Ngram):
    def __init__(self,db):
        self.key_pos = 3
        self.search_pos = 1
        self.left = str(conf['search_aobjects_left'])
        self.right = str(conf['search_aobjects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.name = 'adj-object-clusters'
        self.column = 10
        self.global_limit = conf['cluster_maxrows']
        super(Ngram,self).__init__(db)

class ObjectAdjs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 3
        self.left = str(conf['search_aobjects_right'])
        self.right = str(conf['search_aobjects_left'])
        self.mi = str(conf['verb_min_MI'])
        self.column = 10
        self.global_limit = conf['candidates_maxrows']
        self.name = 'aobject-candidates'
        super(Ngram,self).__init__(db)

class VerbSubjects(Ngram):
    def __init__(self,db):
        self.key_pos = 2
        self.search_pos = 1
        self.left = str(conf['search_subjects_left'])
        self.right = str(conf['search_subjects_right'])
        self.mi = str(conf['noun_min_MI'])
        self.column = 10
        self.global_limit = conf['cluster_maxrows']
        self.name = 'subject-clusters'
        super(Ngram,self).__init__(db)

class ObjectVerbs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 2
        self.left = str(conf['search_object_verbs_left'])
        self.right = str(conf['search_object_verbs_right'])
        self.mi = str(conf['verb_min_MI'])
        self.column = 10
        self.global_limit = conf['candidates_maxrows']
        self.name = 'object-candidates'
        super(Ngram,self).__init__(db)

class SubjectVerbs(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 2
        self.left = str(conf['search_subject_verbs_left'])
        self.right = str(conf['search_subject_verbs_right'])
        self.mi = str(conf['verb_min_MI'])
        self.column = 10
        self.global_limit = conf['candidates_maxrows']
        self.name = 'subject-candidates'
        super(Ngram,self).__init__(db)

class NounNoun(Ngram):
    def __init__(self,db):
        self.key_pos = 1
        self.search_pos = 1
        self.left = str(conf['neuman_graph']['nn_window'])
        self.right = self.left
        self.mi = str(conf['noun_min_MI'])
        self.column = 10
        self.global_limit = conf['cluster_maxrows'] 
        self.name = 'nn-cluster'
        super(Ngram,self).__init__(db)

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
        out.write(time.ctime(self.timestamp)+"\n")
        out.write(self.typ+" "+self.note+"\n")
        out.write("parameters:\n"+printdict(self.params,True))
        out.write("\nNeuman Score: "+self.neuman_eval())
        out.write("\nMRR: "+self.evaluate())
        return out.getvalue()

    def __repr__(self):
        return self.__str__()
    
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
        return self.data['score'].mean()

    def neuman_eval(self):
        frac = self.data['neuman_score'].sum()/self.howmany()
        return "{:.1%}".format(frac)

if __name__ == '__main__':
    print("to explore the cache type data=explore_cache()")
