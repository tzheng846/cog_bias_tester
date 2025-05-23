#!/usr/bin/env python3
"""
Cognitive Bias Quiz with HMM Inference (condensed)

- 10 narrative questions.
- Log-space Viterbi HMM.
- Summary: dominant style, log-prob, joint-prob, per-question probabilities.
"""

import math

# States & names
states = ['BBN','BAY','ANC','REP','OVC','AVL']
state_full = {
  'BBN':'Base-Rate Neglect','BAY':'Rational Bayesian','ANC':'Anchoring',
  'REP':'Representativeness','OVC':'Overconfidence','AVL':'Availability'
}

# Observations
obs_labels = ['Over','Corr','Anch','Rep','Overconf','Avail','Rand']

# HMM parameters
pi = {'BBN':0.25,'BAY':0.20,'ANC':0.15,'REP':0.15,'OVC':0.15,'AVL':0.10}
A = {
  'BBN':{'BBN':0.60,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.05},
  'BAY':{'BBN':0.10,'BAY':0.70,'ANC':0.05,'REP':0.05,'OVC':0.05,'AVL':0.05},
  'ANC':{'BBN':0.05,'BAY':0.15,'ANC':0.60,'REP':0.05,'OVC':0.05,'AVL':0.10},
  'REP':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.60,'OVC':0.10,'AVL':0.05},
  'OVC':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.60,'AVL':0.10},
  'AVL':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.60}
}
B = {
  'BBN':{'Over':0.80,'Corr':0.15,'Anch':0.00,'Rep':0.00,'Overconf':0.00,'Avail':0.00,'Rand':0.05},
  'BAY':{'Over':0.02,'Corr':0.90,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.00},
  'ANC':{'Over':0.05,'Corr':0.15,'Anch':0.70,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
  'REP':{'Over':0.05,'Corr':0.15,'Anch':0.02,'Rep':0.70,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
  'OVC':{'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.70,'Avail':0.02,'Rand':0.09},
  'AVL':{'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.70,'Rand':0.09}
}

# Questions
questions = [
  {"text":"1) 1,000 ppl; 10 cases; test flags 9/10 cases & 99 healthy. Mia tests +. More likely?",
   "options":{'A':"She has it",'B':"False alarm"},"map":{'A':'Over','B':'Corr'}},
  {"text":"2) Coin: HHHHHH. Bet tails, heads, or no bet?",
   "options":{'A':"Tails",'B':"Heads",'C':"No bet"},"map":{'A':'Rep','B':'Anch','C':'Corr'}},
  {"text":"3) 2% have tattoo; witness IDs. More likely?",
   "options":{'A':"Correct",'B':"Mistaken"},"map":{'A':'Over','B':'Corr'}},
  {"text":"4) You asked $90K; offer $70K. Counter-offer?",
   "options":{'A':"$90K",'B':"$75K",'C':"$70K"},"map":{'A':'Anch','B':'Corr','C':'Rand'}},
  {"text":"5) Clinic: 1/20 have condition; test 9/10 & misflags 1/10 healthy. You +. More likely?",
   "options":{'A':"Have it",'B':"False positive"},"map":{'A':'Over','B':'Corr'}},
  {"text":"6) Sam quiet & loves puzzles; firm mostly lawyers, some engineers. Sam is?",
   "options":{'A':"Engineer",'B':"Lawyer"},"map":{'A':'Rep','B':'Corr'}},
  {"text":"7) Which scares more: plane crash or drowning?",
   "options":{'A':"Plane crash",'B':"Drowning"},"map":{'A':'Avail','B':'Corr'}},
  {"text":"8) Which kills more yearly: crash or drowning?",
   "options":{'A':"Plane crash",'B':"Drowning"},"map":{'A':'Avail','B':'Corr'}},
  {"text":"9) Linda: teller vs teller+activist. More likely?",
   "options":{'A':"Teller",'B':"Teller+activist"},"map":{'A':'Corr','B':'Rep'}},
  {"text":"10) Test prior20%; sens80%, spec90%. You +. More likely?",
   "options":{'A':"You have it",'B':"False alarm"},"map":{'A':'Corr','B':'Over'}}
]

def viterbi_log(obs):
    eps=1e-6
    for s in states:
        for o in obs_labels:
            B[s][o]=B[s].get(o,eps) or eps
    logpi={s:math.log(pi[s]) for s in states}
    logA={s:{t:math.log(A[s][t]) for t in states} for s in states}
    logB={s:{o:math.log(B[s][o]) for o in obs_labels} for s in states}
    V=[{s:logpi[s]+logB[s][obs[0]] for s in states}]
    path={s:[s] for s in states}
    for ot in obs[1:]:
        prevV=V[-1]; curr={}; newpath={}
        for cur in states:
            p,lp=max(((p,prevV[p]+logA[p][cur]+logB[cur][ot]) for p in states), key=lambda x:x[1])
            curr[cur]=lp; newpath[cur]=path[p]+[cur]
        V.append(curr); path=newpath
    best=max(states,key=lambda s:V[-1][s]); return V[-1][best], path[best], V

def main():
    print("=== Cognitive Bias Quiz ===")
    answers=[]
    for q in questions:
        print("\n"+q['text'])
        for k,v in q['options'].items(): print(f" {k}) {v}")
        ans=input("Choice: ").strip().upper()
        while ans not in q['options']: ans=input(f"Choose {','.join(q['options'])}: ").strip().upper()
        answers.append(ans)
    obs=[q['map'][a] for q,a in zip(questions,answers)]
    logp,path,V=viterbi_log(obs)
    counts={s:path.count(s) for s in states}; dom=max(counts,key=counts.get)
    print(f"\nDominant style: {state_full[dom]} ({counts[dom]}/{len(obs)})")
    print(f"Log-prob: {logp:.2f}, Joint-prob: {math.exp(logp)}\n")
    print("Per-question probabilities:")
    for i,ot in enumerate(obs,1):
        logs=V[i-1]; m=max(logs.values())
        exps={s:math.exp(logs[s]-m) for s in states}; tot=sum(exps.values())
        probs={s:exps[s]/tot for s in states}
        print(f"Q{i}({ot}): "+", ".join(f"{state_full[s]}={probs[s]:.2f}" for s in states))

if __name__=="__main__":
    main()
