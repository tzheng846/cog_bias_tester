#!/usr/bin/env python3
"""
Narrative Cognitive Bias Quiz with HMM Inference and Log-Probability Explanation

- Hidden states: Base-Rate Neglect, Rational Bayesian, Anchoring,
  Representativeness, Overconfidence, Availability
- Observation labels: Over, Corr, Anch, Rep, Overconf, Avail, Rand

The script:
1. Presents 10 narrative questions probing different biases.
2. Maps each answer into one of the observation labels.
3. Uses a log-space Viterbi algorithm (with smoothing) to infer the most likely bias state sequence.
4. Prints:
   - Your dominant reasoning style.
   - How many times each style was most likely.
   - A per-question table of inferred probabilities.
   - The **log-probability** of the inferred path and its interpretation.
"""

import math

# 1) Hidden states and display names
states = ['BBN','BAY','ANC','REP','OVC','AVL']
state_full = {
    'BBN': 'Base-Rate Neglect',
    'BAY': 'Rational Bayesian',
    'ANC': 'Anchoring',
    'REP': 'Representativeness',
    'OVC': 'Overconfidence',
    'AVL': 'Availability'
}

# 2) Observation labels
obs_labels = ['Over','Corr','Anch','Rep','Overconf','Avail','Rand']

# 3) HMM parameters (expert-elicited)
pi =   {'BBN':0.25,'BAY':0.20,'ANC':0.15,'REP':0.15,'OVC':0.15,'AVL':0.10}
A = {
  'BBN':{'BBN':0.60,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.05},
  'BAY':{'BBN':0.10,'BAY':0.70,'ANC':0.05,'REP':0.05,'OVC':0.05,'AVL':0.05},
  'ANC':{'BBN':0.05,'BAY':0.15,'ANC':0.60,'REP':0.05,'OVC':0.05,'AVL':0.10},
  'REP':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.60,'OVC':0.10,'AVL':0.05},
  'OVC':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.60,'AVL':0.10},
  'AVL':{'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.60},
}
B = {
  'BBN':{'Over':0.80,'Corr':0.15,'Anch':0.00,'Rep':0.00,'Overconf':0.00,'Avail':0.00,'Rand':0.05},
  'BAY':{'Over':0.02,'Corr':0.90,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.00},
  'ANC':{'Over':0.05,'Corr':0.15,'Anch':0.70,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
  'REP':{'Over':0.05,'Corr':0.15,'Anch':0.02,'Rep':0.70,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
  'OVC':{'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.70,'Avail':0.02,'Rand':0.09},
  'AVL':{'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.70,'Rand':0.09},
}

# 4) Narrative questions with option→observation maps
questions = [
    {
        'text': "1) Out of 1,000 people, 10 have a disease. A test catches 9 true cases but wrongly flags 99 healthy. Mia tests positive. Which is more likely?",
        'options': {'A': "She truly has the disease", 'B': "It's a false alarm"},
        'map':     {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "2) A coin lands heads six times in a row. Do you bet on tails, heads, or no bet?",
        'options': {'A': "Tails (gambler's fallacy)", 'B': "Heads (stick with pattern)", 
                    'C': "No bet (50/50)"},
        'map':     {'A': 'Rep', 'B': 'Anch', 'C': 'Corr'}
    },
    {
        'text': "3) Only 2% have a rare tattoo. A witness IDs someone with that tattoo. Which is more likely?",
        'options': {'A': "Witness is correct", 'B': "Witness is mistaken"},
        'map':     {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "4) You mentioned $90K, company offers $70K. How do you counter-offer?",
        'options': {'A': "Ask $90K", 'B': "Suggest $75K", 'C': "Accept $70K"},
        'map':     {'A': 'Anch', 'B': 'Corr', 'C': 'Rand'}
    },
    {
        'text': "5) In clinic, 1 in 20 have a condition. Test catches 9/10 but misflags 1/10 healthy. You test positive. Which is more likely?",
        'options': {'A': "You have it", 'B': "False positive"},
        'map':     {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "6) Sam is quiet and loves puzzles. In a firm with mostly lawyers but some engineers, who's Sam more likely to be?",
        'options': {'A': "Engineer", 'B': "Lawyer"},
        'map':     {'A': 'Rep', 'B': 'Corr'}
    },
    {
        'text': "7) Which scares you more when traveling: plane crash or drowning?",
        'options': {'A': "Plane crash", 'B': "Drowning"},
        'map':     {'A': 'Avail', 'B': 'Corr'}
    },
    {
        'text': "8) Which actually causes more deaths per year: plane crash or drowning?",
        'options': {'A': "Plane crash", 'B': "Drowning"},
        'map':     {'A': 'Avail', 'B': 'Corr'}
    },
    {
        'text': "9) Linda is a bank teller vs bank teller & activist. Which is more likely?",
        'options': {'A': "Bank teller", 'B': "Bank teller & activist"},
        'map':     {'A': 'Corr', 'B': 'Rep'}
    },
    {
        'text': "10) Test prior 20%, sens 80%, spec 90%. You test positive. Which is more likely?",
        'options': {'A': "You have it", 'B': "False alarm"},
        'map':     {'A': 'Corr', 'B': 'Over'}
    },
]

# 5) Log-space Viterbi with smoothing
def viterbi_log(obs_seq):
    eps = 1e-6
    for s in states:
        for o in obs_labels:
            if B[s].get(o,0)==0:
                B[s][o]=eps
    logpi = {s:math.log(pi[s]) for s in states}
    logA  = {s:{t:math.log(A[s][t]) for t in states} for s in states}
    logB  = {s:{o:math.log(B[s][o]) for o in obs_labels} for s in states}

    V = [{}]; path = {}
    # Initialization
    for s in states:
        V[0][s] = logpi[s] + logB[s][obs_seq[0]]
        path[s] = [s]
    # Recursion
    for t in range(1, len(obs_seq)):
        V.append({})
        newpath = {}
        for cur in states:
            prev, lp = max(
                ((p, V[t-1][p] + logA[p][cur] + logB[cur][obs_seq[t]]) for p in states),
                key=lambda x: x[1]
            )
            V[t][cur]    = lp
            newpath[cur] = path[prev] + [cur]
        path = newpath
    # Termination
    last = len(obs_seq)-1
    best_state = max(states, key=lambda s: V[last][s])
    final_logp = V[last][best_state]
    return final_logp, path[best_state], V

# 6) Run quiz
def run_quiz():
    answers = []
    print("\n=== Cognitive Bias Quiz ===\n")
    for q in questions:
        print(q['text'])
        for k,v in q['options'].items():
            print(f"  {k}) {v}")
        choice = input("Choice: ").strip().upper()
        while choice not in q['options']:
            choice = input("Choose one of " + "/".join(q['options']) + ": ").strip().upper()
        answers.append(choice)
        print()
    obs_seq = [q['map'][a] for q,a in zip(questions, answers)]
    print("Observations:", obs_seq)

    logp, path, V = viterbi_log(obs_seq)
    counts = {s: path.count(s) for s in states}
    dominant = max(counts, key=counts.get)

    # Summary
    print("\n=== Summary ===")
    print(f"Dominant style: {state_full[dominant]} "
          f"({counts[dominant]}/{len(obs_seq)})")
    print(f"Log-Probability of this inferred path: {logp:.2f}")
    # Interpret log-prob
    prob = math.exp(logp)
    print(f"Approximate joint probability of your sequence: {prob}")
    print("\n(Higher values mean the model finds your reasoning sequence more likely under that bias profile.)\n")

    # Per-question breakdown
    print("Per-question reasoning probabilities:")
    for i, obs in enumerate(obs_seq, 1):
        logs = V[i-1]
        m = max(logs.values())
        exps = {s: math.exp(logs[s]-m) for s in states}
        total = sum(exps.values())
        probs = {s: exps[s]/total for s in states}
        print(f"Q{i} ({obs}):", end=" ")
        for s in states:
            print(f"{state_full[s]}={probs[s]:.2f}", end="; ")
        print(f"-> {state_full[max(probs, key=probs.get)]}")

if __name__ == "__main__":
    run_quiz()

