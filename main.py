import streamlit as st
import math
import pandas as pd

# 1) Hidden states and display names
states = ['BBN','BAY','ANC','REP','OVC','AVL']
state_full = {
    'BBN':'Base-Rate Neglect',
    'BAY':'Rational Bayesian',
    'ANC':'Anchoring',
    'REP':'Representativeness',
    'OVC':'Overconfidence',
    'AVL':'Availability'
}
obs_labels = ['Over','Corr','Anch','Rep','Overconf','Avail','Rand']

# 2) HMM parameters (same as before)
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

# 3) Mixed‐type question bank
questions = [
    # Numeric input questions (type='num')
    {
      'type':'num',
      'text': "Q1) Disease prevalence = 1%; test sensitivity=90%, specificity=90%. You test positive. Enter P(disease) (%)",
      'correct':  (0.9*0.01)/(0.9*0.01 + 0.1*0.99)*100,  # ~8.3
      'anchor':   90.0,
      'tol':      5.0
    },
    {
      'type':'num',
      'text': "Q5) Disease prevalence = 5%; test sensitivity=90%, specificity=90%. You test positive. Enter P(disease) (%)",
      'correct':  (0.9*0.05)/(0.9*0.05 + 0.1*0.95)*100,  # ~32.1
      'anchor':   90.0,
      'tol':      5.0
    },
    # Multiple‐choice questions
    {
      'type':'mc',
      'text':"Q2) Coin flips HHHHHH → next? (Tails vs Heads vs 50/50 vs Unsure)",
      'options':{'A':"Tails","B":"Heads","C":"50/50","D":"Unsure"},
      'map':     {'A':'Rep','B':'Anch','C':'Corr','D':'Rand'}
    },
    {
      'type':'mc',
      'text':"Q3) Witness IDs a 2%-profile suspect. Guilty %",
      'options':{'A':"80–90%","B":"15–20%","C":"98%","D":"Unsure"},
      'map':     {'A':'Over','B':'Corr','C':'Anch','D':'Rand'}
    },
    {
      'type':'mc',
      'text':"Q4) Job offer $70k vs expected $90k. Fair salary?",
      'options':{'A':"$80k","B":"$90k","C":"Unsure"},
      'map':     {'A':'Corr','B':'Anch','C':'Rand'}
    },
    {
      'type':'mc',
      'text':"Q6) 70 lawyers & 30 engineers; Sam loves puzzles. Profession?",
      'options':{'A':"Engineer","B":"Lawyer","C":"Unsure"},
      'map':     {'A':'Rep','B':'Corr','C':'Rand'}
    },
    {
      'type':'mc',
      'text':"Q7) Pick the most vivid example: plane crash vs auto accident vs equal vs unsure",
      'options':{'A':"Plane crash","B":"Auto accident","C":"Equal","D":"Unsure"},
      'map':     {'A':'Avail','B':'Corr','C':'Rand','D':'Rand'}
    },
    # New numeric questions
    {
      'type':'num',
      'text': "Q13) Expected value of a fair six-sided die. Enter your answer",
      'correct': 3.5,
      'anchor': None,
      'tol': 0.5
    },
    {
      'type':'num',
      'text': "Q14) A test has prior 20%, sensitivity 80%, specificity 90%. Positive result: Enter P(disease) (%)",
      'correct': (0.8*0.2)/(0.8*0.2 + 0.1*0.8)*100,  # ~66.7
      'anchor': 90.0,
      'tol': 5.0
    },
]

# 4) Viterbi in log‐space with smoothing
def viterbi_log(obs_seq):
    eps = 1e-6
    for s in states:
        for o in obs_labels:
            if B[s].get(o,0)==0:
                B[s][o]=eps
    logpi = {s:math.log(pi[s]) for s in states}
    logA  = {s:{t:math.log(A[s][t]) for t in states} for s in states}
    logB  = {s:{o:math.log(B[s][o]) for o in obs_labels} for s in states}

    V=[{}]; path={}
    for s in states:
        V[0][s]=logpi[s] + logB[s][obs_seq[0]]
        path[s]=[s]
    for t in range(1,len(obs_seq)):
        V.append({}); newpath={}
        ot = obs_seq[t]
        for cur in states:
            prev,lp = max(
                ((p, V[t-1][p] + logA[p][cur] + logB[cur][ot]) for p in states),
                key=lambda x:x[1]
            )
            V[t][cur]=lp
            newpath[cur]=path[prev]+[cur]
        path = newpath
    last = len(obs_seq)-1
    best = max(states, key=lambda s: V[last][s])
    return V[last][best], path[best], V

# 5) Mapping numeric responses
def map_numeric(resp, correct, anchor, tol):
    if abs(resp - correct) <= tol:
        return 'Corr'
    if anchor is not None and abs(resp - anchor) <= tol:
        return 'Anch'
    if resp > correct + tol:
        return 'Over'
    return 'Rand'

# 6) Streamlit UI
st.title("🧠 Cognitive Bias Simulator")

with st.form("quiz"):
    raw_answers = []
    for q in questions:
        if q['type']=='mc':
            ans = st.radio(q['text'], list(q['options'].keys()),
                           format_func=lambda k,opts=q['options']: opts[k])
            raw_answers.append(ans)
        else:  # numeric
            val = st.number_input(q['text'], value=0.0, format="%.2f")
            raw_answers.append(val)
    submitted = st.form_submit_button("Submit")

if submitted:
    # Map to obs_seq
    obs_seq = []
    for q, ra in zip(questions, raw_answers):
        if q['type']=='mc':
            obs_seq.append(q['map'][ra])
        else:
            obs_seq.append(map_numeric(ra, q['correct'], q['anchor'], q['tol']))

    # Infer HMM
    logp, path, V = viterbi_log(obs_seq)
    counts = {s:path.count(s) for s in states}
    dominant = max(counts, key=counts.get)

    # Summary
    st.header("Your Results")
    st.markdown(f"**Dominant reasoning style:** {state_full[dominant]}  \n"
                f"You used it in {counts[dominant]} of {len(obs_seq)} questions.  \n"
                f"Log-probability: {logp:.1f}")

    # Bar chart
    df_counts = pd.DataFrame({state_full[s]:counts[s] for s in states}, index=["Count"])
    st.bar_chart(df_counts.T)

    # Per-question table
    rows = []
    for t, obs in enumerate(obs_seq):
        logs = V[t]
        m = max(logs.values())
        exps = {s:math.exp(logs[s]-m) for s in states}
        tot = sum(exps.values())
        probs = {state_full[s]: exps[s]/tot for s in states}
        probs["Question"] = f"Q{t+1}"
        rows.append(probs)
    df = pd.DataFrame(rows).set_index("Question")
    st.subheader("Per-Question Reasoning Probabilities")
    st.dataframe(df.style.format("{:.1%}"))

    st.write("Numbers within tolerance of the correct answer are ‘Correct’; above that are ‘Over’ or ‘Anch’ when matching test accuracy; otherwise ‘Rand.’")
