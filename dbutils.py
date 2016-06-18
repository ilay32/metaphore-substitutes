import pandas,sys,os,yaml,pymssql,math,copy,pickle,json,numbers,six,io,time,threading
import numpy as np
from scipy.spatial.distance import cosine
from datetime import datetime
conf = yaml.load(open('params.yml').read())

class DB:
    def __init__(self,catalog=None):
        specs = yaml.load(open('db.yml').read())
        self.server_name = specs['server']
        self.user = specs['user']
        self.password = specs['pass']
        self.catalog = catalog
        self.abst = dict()
        self.vecs = dict()
        self.conn = pymssql.connect(self.server_name, self.user, self.password,self.catalog)
    
    def query(self,q):
        c = self.conn.cursor()
        c.setoutputsize(conf['cluster_maxrows'] + conf['candidates_maxrows'])
        try:
            c.execute(q)
        except pymssql.DatabaseError as e: 
            print("trying to execute ",q," got ",e)
            return None
        return c 
    
    def destroy(self):
        self.conn.close()

class SingleWordData:
    dbcache = 'dbcache'
    def __init__(self,db):
        self.db = db
        self.table_path = self.get_table_path()
        self.table = self.get_table()
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
        return len(obj) ==  0

    def get(self,word):
        word = word.replace("'","")
        if word in self.notfound:
            print("this shouldn't be here")
        if word in self.table:
            return self.table[word]
        else:
            q = self.db.query(self.queryscheme(word))
            qr = self.handlequery(q)
            if not SingleWordData.empty(qr):
                self.table[word] = qr
            q = None
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
        with open(self.table_path, 'wb') as f:
            pickle.dump(self.table,f)
    
    def destroy(self,save=False):
        self.save_table()
        self.db.destroy()


class Abst(SingleWordData):
    def queryscheme(self,word):
        return "SELECT ABSTRACT_SCALE FROM PHRASE_ABSTRACT WHERE PHRASE='{0}'".format(word.lower())

    def handlequery(self,q):
        try: 
            f = q.fetchall()
            if not q or len(f) == 0:
                return 0
            return f[0][0]
        except:
            return 0
    
    def search_table(self):
        return 'abst'

class Vecs(SingleWordData):
    def queryscheme(self,word):
        return "GetRows '{0}'".format(word.lower())

    def handlequery(self,query):
        try:
            ret = dict()
            for row in query:
                ret.update({row[2] : row[3]})
            return ret
        except:
            print("db error for ",query)
            return None
   
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

class Lex(SingleWordData):
    
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
    
    def ready(self,word):
        query_base = "'{0}',{1},{2},{3},{4},{5}".format(
            word,
            self.key_pos, # this is the POS type of verb
            self.search_pos,
            self.left, # how many words to look behind
            self.right, # how many words to look ahead
            self.mi # minimal mutual information required for results
        )
        self.query_base = query_base
        return bool(self.db.query("IsNgramsReady "+query_base+",1")) == True
    
    def queryscheme(self,word):
        cv = threading.Condition
        while not self.ready(word): 
            cv.wait() 
        return "GetNgrams "+self.query_base    

    def handlequery(self,q):
        try:
            ret = set()
            cur = 0
            rows = q.fetchall()
            while len(ret) < self.global_limit and cur < len(rows):
                ret.add(rows[cur][self.column])
                cur += 1
            return list(ret)
        except:
            print("db error for ",q)
            return None

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

#
# general utility functions
def printlist(l,k=5,silent=False):
    i = 0
    last = min(k,len(l))
    out = io.StringIO()
    while i < last:
        if isinstance(l[i],(numbers.Number,six.string_types,tuple)):
            out.write(str(l[i]))
        elif isinstance(l[i],list):
            out.write(printlist(min(1,k-1),len(l[i]),True))
        elif isinstance(l[i],dict):
            out.write(printdict(l[i]))
        if i == last and i < len(l):
            ret += "...("+str(len(l))+")"
        if i < last - 1:
            out.write(",")
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
        print(i+1,".\t",os.path.basename(cache))
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
    
    def __init__(self,df,note):
        self.data = df
        self.params = yaml.load(open('params.yml').read())
        self.timestamp = int(time.time())
        self.note = note.strip()
    
    def __str__(self):
        out = io.StringIO()
        out.write(time.ctime(self.timestamp)+"\n")
        out.write(self.note+"\n")
        out.write("parameters:\n"+printdict(self.params,True))
        return out.getvalue()

    def __repr__(self):
        return self.__str__()
    
    def tofile(self,location=None):
        if location != None and not os.path.isdir(location):
            print(location," is not a valid directory. aborting")
            return
        filename  = self.note.replace(' ','_')
        path =  os.path.join(location,filename) if location != None else filename
        with open(os.path.join(path+".txt"),"w") as f:
            d = self.data
            f.write(str(self)+"\n")
            for i,row in d.iterrows():
                f.write(row.verb+" "+row.noun+" ("+row.rel+")")
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

    def show(self,verb=None,noun=None):
        d = self.data
        if verb == 'pairs':
            for i,row in d.iterrows():
                print(str(i)+".",row.verb," ",row.noun," ",row.rel,"\n")
            return

        if verb in ['no_vector','no_abst','ignored']:
            acc = set()
            for l in range(len(d)):
                acc = acc.union(set(d.loc[l,verb]))
            ans = list(acc)
            printlist(ans,len(ans))
            return
        
        pairs = list()
        if verb!=None and noun!=None:
            pairs = d[d.verb == verb & d.noun == noun]
        elif verb != None and noun == None:
            pairs = d[d.verb == verb]
        elif noun != None and verb == None:
            pairs =  d[d.noun == noun]
                    
        else:
            pairs = d
        for i,row in pairs.iterrows():
            printlist(row['substitutes'],self.params['number_of_candidates']) 
    
