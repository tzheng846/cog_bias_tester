import streamlit as st
import math
import pandas as pd

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

# 3) HMM parameters
pi = {'BBN':0.25,'BAY':0.20,'ANC':0.15,'REP':0.15,'OVC':0.15,'AVL':0.10}
A = {
    'BBN': {'BBN':0.60,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.05},
    'BAY': {'BBN':0.10,'BAY':0.70,'ANC':0.05,'REP':0.05,'OVC':0.05,'AVL':0.05},
    'ANC': {'BBN':0.05,'BAY':0.15,'ANC':0.60,'REP':0.05,'OVC':0.05,'AVL':0.10},
    'REP': {'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.60,'OVC':0.10,'AVL':0.05},
    'OVC': {'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.60,'AVL':0.10},
    'AVL': {'BBN':0.05,'BAY':0.15,'ANC':0.05,'REP':0.05,'OVC':0.10,'AVL':0.60}
}
B = {
    'BBN': {'Over':0.80,'Corr':0.15,'Anch':0.00,'Rep':0.00,'Overconf':0.00,'Avail':0.00,'Rand':0.05},
    'BAY': {'Over':0.02,'Corr':0.90,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.00},
    'ANC': {'Over':0.05,'Corr':0.15,'Anch':0.70,'Rep':0.02,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
    'REP': {'Over':0.05,'Corr':0.15,'Anch':0.02,'Rep':0.70,'Overconf':0.02,'Avail':0.02,'Rand':0.04},
    'OVC': {'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.70,'Avail':0.02,'Rand':0.09},
    'AVL': {'Over':0.05,'Corr':0.10,'Anch':0.02,'Rep':0.02,'Overconf':0.02,'Avail':0.70,'Rand':0.09}
}

# 4) Narrative questions with option→observation mappings
questions = [
    {
        'text': "1. In a town of 1,000 people, 10 have a disease. A test catches 9 true cases but wrongly flags 99 healthy individuals. You test positive. Which is more likely?",
        'options': {'A': "You have the disease.", 'B': "It's a false positive."},
        'map': {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "2. You observe a fair coin landing heads six times in a row. Do you bet on tails, bet on heads, or refrain because it remains 50/50?",
        'options': {'A': "Bet on tails.", 'B': "Bet on heads.", 'C': "Refrain from betting."},
        'map': {'A': 'Rep', 'B': 'Anch', 'C': 'Corr'}
    },
    {
        'text': "3. Only 2% of people have a rare tattoo. A witness identifies someone with that tattoo as a suspect. Which is more likely?",
        'options': {'A': "The witness is correct.", 'B': "The witness is mistaken."},
        'map': {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "4. During salary negotiation, you mention $90,000 but receive an offer of $70,000. How do you respond?",
        'options': {'A': "Insist on $90,000.", 'B': "Propose $75,000.", 'C': "Accept $70,000."},
        'map': {'A': 'Anch', 'B': 'Corr', 'C': 'Rand'}
    },
    {
        'text': "5. At a clinic, 1 in 20 patients has a condition. A test detects 9 out of 10 true cases but misclassifies 1 out of 10 healthy patients as positive. You test positive. Which is more likely?",
        'options': {'A': "You have the condition.", 'B': "It's a false positive."},
        'map': {'A': 'Over', 'B': 'Corr'}
    },
    {
        'text': "6. Sam is quiet, detail-oriented, and enjoys puzzles. In a company with mostly lawyers and some engineers, which profession is Sam more likely to have?",
        'options': {'A': "Engineer.", 'B': "Lawyer."},
        'map': {'A': 'Rep', 'B': 'Corr'}
    },
    {
        'text': "7. Which scenario would you find more frightening when planning a vacation: a plane crash or drowning?",
        'options': {'A': "Plane crash.", 'B': "Drowning."},
        'map': {'A': 'Avail', 'B': 'Corr'}
    },
    {
        'text': "8. Which event actually causes more deaths per year: plane crashes or drownings?",
        'options': {'A': "Plane crashes.", 'B': "Drownings."},
        'map': {'A': 'Avail', 'B': 'Corr'}
    },
    {
        'text': "9. Linda is a bank teller and active in social justice causes. Which is more likely?",
        'options': {'A': "She is a bank teller.", 'B': "She is a bank teller and an activist."},
        'map': {'A': 'Corr', 'B': 'Rep'}
    },
    {
        'text': "10. A test for a disease has 20% prevalence, 80% sensitivity, and 90% specificity. You test positive. Which is more likely?",
        'options': {'A': "You have the disease.", 'B': "It's a false positive."},
        'map': {'A': 'Corr', 'B': 'Over'}
    }
]

# 5) Log-space Viterbi with smoothing
@st.cache
def build_hmm():
    eps = 1e-6
    # Ensure every emission exists
    for s in states:
        for o in obs_labels:
            B[s].setdefault(o, eps)
            if B[s][o] == 0: B[s][o] = eps
    logpi = {s: math.log(pi[s]) for s in states}
    logA = {s: {t: math.log(A[s][t]) for t in states} for s in states}
    logB = {s: {o: math.log(B[s][o]) for o in obs_labels} for s in states}
    return logpi, logA, logB

def viterbi_log(obs_seq, logpi, logA, logB):
    V = [{s: logpi[s] + logB[s][obs_seq[0]] for s in states}]
    path = {s: [s] for s in states}
    for ot in obs_seq[1:]:
        prev = V[-1]
        curr = {}
        newp = {}
        for cur in states:
            p, lp = max(
                ((p, prev[p] + logA[p][cur] + logB[cur][ot]) for p in states),
                key=lambda x: x[1]
            )
            curr[cur] = lp
            newp[cur] = path[p] + [cur]
        V.append(curr)
        path = newp
    best = max(states, key=lambda s: V[-1][s])
    return V[-1][best], path[best], V

# 6) Streamlit UI
st.title("Cognitive Bias Quiz")
st.write("Answer the questions below to discover your dominant reasoning style.")

with st.form("quiz"):
    answers = []
    for q in questions:
        ans = st.radio(q['text'], list(q['options'].keys()),
                       format_func=lambda k, opts=q['options']: opts[k])
        answers.append(ans)
    submitted = st.form_submit_button("Submit")

if submitted:
    obs_seq = [q['map'][a] for q, a in zip(questions, answers)]
    logpi, logA, logB = build_hmm()
    final_logp, state_path, V = viterbi_log(obs_seq, logpi, logA, logB)
    counts = {s: state_path.count(s) for s in states}
    dominant = max(counts, key=counts.get)

    st.subheader("Results")
    st.write(f"**Dominant reasoning style:** {state_full[dominant]} ({counts[dominant]}/{len(obs_seq)})")
    st.write(f"**Log-probability:** {final_logp:.2f}")
    st.write(f"**Joint probability:** {math.exp(final_logp):.6f}")

    df_counts = pd.DataFrame({state_full[s]: counts[s] for s in states}, index=["Count"]).T
    st.bar_chart(df_counts)

    rows = []
    for i, ot in enumerate(obs_seq):
        logs = V[i]
        m = max(logs.values())
        exps = {s: math.exp(logs[s] - m) for s in states}
        tot = sum(exps.values())
        probs = {state_full[s]: exps[s]/tot for s in states}
        probs["Question"] = f"Q{i+1}"
        rows.append(probs)
    df_probs = pd.DataFrame(rows).set_index("Question")
    st.subheader("Per-Question Probabilities")
    st.dataframe(df_probs.style.format("{:.1%}"))

