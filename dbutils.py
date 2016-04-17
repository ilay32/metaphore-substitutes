import pandas,sys,os,json,pymssql
import numpy as np
from scipy.spatial.distance import cosine
from MetaphorResearch.DAL_AbstractionDB.DbAccess import ServerConnection

class DB:
    def __init__(self,catalog=None):
        specs = json.load(open('MetaphorResearch/user.spec'))
        self.server_name = specs['server']
        self.user = specs['user']
        self.password = specs['pass']
        self.catalog = catalog
        self.abst = dict()
        self.vecs = dict()
        self.conn = pymssql.connect(self.server_name, self.user, self.password,self.catalog)
    
    def query(self,q):
        c = self.conn.cursor()
        c.execute(q)
        return c 

class SingleWordData:
    def __init__(self,db):
        self.db = db
        self.table = dict()
        self.notfound = set()
    
    def has(self,word):
        if word not in self.notfound:
            self.get(word)
            if word in self.table:
                return True
            self.notfound.add(word)
        return False
    
    def empty(obj):
        if isinstance(obj,int) or  isinstance(obj,float):
            return obj == 0
        return len(obj) == 0

    def get(self,word):
        if word in self.notfound:
            print("this shouldn't be here")
        if word in self.table:
            return self.table[word]
        else:
            q = self.db.query(self.queryscheme(word))
            qr = self.handlequery(q)
            if not SingleWordData.empty(qr):
                self.table[word] = qr
        return qr
    def __str__(self):
        return self.table[:min(len(self.table),5)]

class Abst(SingleWordData):
    def queryscheme(self,word):
        return "SELECT ABSTRACT_SCALE FROM PHRASE_ABSTRACT WHERE PHRASE='{0}'".format(word.lower())

    def handlequery(self,q):
        f = q.fetchall()
        if not q or len(f) == 0:
            return 0
        return f[0][0]

class Vecs(SingleWordData):
    def queryscheme(self,word):
        return "GetRows '{0}'".format(word.lower())

    def handlequery(self,query):
        ret = []
        for row in query:
            ret.append(row[3])
        return np.array(ret)
   
    # these methods expect np arrays
    def distance(self,u,v):
        m = min(len(u),len(v))
        return cosine(u[:m],v[:m])
    
    def addition(self,u,v):
        m = min(len(u),len(v))
        x = u[:m] + v[:m]
        return x

class Lex(SingleWordData):
    
    def handlequery(self,q):
        ret = tuple()
        for r in q:
            ret += (r[0],)
        return ret
    
    def queryscheme(self,word):
        return "SELECT DISTINCT posType FROM Lexicon WHERE word='{0}' AND lemma = '{0}'".format(word)
def explore(query,catalog):
    specs = json.load(open(os.path.join(os.getcwd(),'MetaphorResearch','user.spec')))
    db = DB(specs,catalog) 
    ans = db.query(query)
    print(ans)
    return
