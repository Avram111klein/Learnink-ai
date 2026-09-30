import json,sys
def lev_ok(T,R):
    n,m=len(T),len(R); D=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0]=i
    for j in range(m+1): D[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1): D[i][j]=min(D[i-1][j-1]+(T[i-1]!=R[j-1]),D[i-1][j]+1,D[i][j-1]+1)
    i,j,ok,conf=n,m,0,[]
    while i>0 or j>0:
        if i>0 and j>0 and D[i][j]==D[i-1][j-1]+(T[i-1]!=R[j-1]):
            if T[i-1]==R[j-1]: ok+=1
            else: conf.append(T[i-1]+'→'+R[j-1])
            i-=1;j-=1
        elif i>0 and D[i][j]==D[i-1][j]+1: conf.append(T[i-1]+'→∅'); i-=1
        else: j-=1
    return ok,conf
def tokens(t):
    out=[];i=0
    while i<len(t):
        c=t[i]
        if c=='^':
            j=i+1
            while j<len(t) and t[j].isdigit(): out.append('^'+t[j]); j+=1
            i=j; continue
        out.append(c); i+=1
    return out
import re
def shape(t): return ((t.count('=')), '/' in t, '^' in t, t.count('(')+t.count(')'), ''.join(sorted(set(re.sub('[^a-z]','',t)))))
def wellformed(t):
    if not t or re.search(r'[+\-=^/.]$',t) or re.match(r'^[+=^/.)]',t) or re.search(r'[+\-=^/.]{2,}',t) or re.search(r'\^\D',t) or re.search(r'\.\D|\D\.',t): return False
    d=0
    for c in t:
        d+= (c=='(')-(c==')')
        if d<0: return False
    return d==0 and not re.search(r'\(\)|\([+=^/.]|[+\-=^/.(]\)',t)
def score(R):
    from collections import Counter
    ok=n=ex=fx=0; C=Counter()
    for r in R:
        T=tokens(r['tgt']); o,c=lev_ok(T,tokens(r['text'])); ok+=o; n+=len(T); C.update(c)
        e=r['text']==r['tgt']; ex+=e
        if not e and r['p']>=0.5 and r['tgt'] not in r['alts'] and wellformed(r['text']) and shape(r['text'])==shape(r['tgt']) and abs(len(r['text'])-len(r['tgt']))<=1: fx+=1
    N=len(R); return dict(n=N,sym=round(ok/n,4),exact=round(ex/N,4),falseX=round(fx/N,4),top=C.most_common(8))
if __name__=='__main__':
    R=json.load(open(sys.argv[1]))
    if isinstance(R[0],list):
        for k,r in enumerate(R): print(k,score(r))
    else: print(score(R))
