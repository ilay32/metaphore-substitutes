import pandas,sys,os,json,pymssql,math,copy
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
    
    def searchval(self,val):
        for w,v in self.table.items():
            if v == val:
                return w

    def has(self,word):
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
        try: 
            f = q.fetchall()
            if not q or len(f) == 0:
                return 0
            return f[0][0]
        except:
            return 0

class Vecs(SingleWordData):
    def queryscheme(self,word):
        return "GetRows '{0}'".format(word.lower())

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
        return -1

    def vectorpair(u,v):
        for col_id in u.keys():
            if col_id in v:
                yield(col_id,u[col_id],v[col_id])
            else:
                yield(col_id,u[col_id],0)
    
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

