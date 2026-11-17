"""Small, transparent BM25 implementation (no retrieval framework)."""
import math, re
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.docs=[tokenize(d) for d in documents]
        self.k1,self.b=k1,b
        self.avgdl=sum(map(len,self.docs))/max(1,len(self.docs))
        self.tf=[Counter(d) for d in self.docs]
        self.df=Counter()
        for terms in self.docs:
            self.df.update(set(terms))
        self.n=len(self.docs)
    def score(self, query):
        q=set(tokenize(query)); scores=[]
        for i,doc in enumerate(self.docs):
            dl=len(doc); s=0.0
            for term in q:
                freq=self.tf[i].get(term,0)
                if not freq: continue
                idf=math.log(1+(self.n-self.df[term]+0.5)/(self.df[term]+0.5))
                denom=freq+self.k1*(1-self.b+self.b*dl/max(1,self.avgdl))
                s += idf*(freq*(self.k1+1))/denom
            scores.append(s)
        return scores
