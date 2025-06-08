import re
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors

# ---- Load and Clean JSON File ----

def load_tickets_from_file(filepath):
    with open(filepath, "r") as f:
        raw_text = f.read()

    # Split by Ticket header and extract valid JSON
    raw_tickets = re.split(r'// Ticket #[0-9]+', raw_text)
    tickets = []

    for block in raw_tickets:
        block = block.strip()
        if block:
            try:
                ticket = json.loads(block)
                tickets.append(ticket)
            except json.JSONDecodeError as e:
                print("Skipping malformed ticket:", e)
    return tickets


# ---- Load Tickets ----

tickets = load_tickets_from_file("ticket_json_files.json")
df = pd.DataFrame(tickets)
print(f"\nLoaded {len(df)} tickets.\n")

# ---- Urgency Classification (1–5 scale) ----

X = df['issue']
y = df['priority']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

print("=== Urgency Classification Report ===")
print(classification_report(y_test, y_pred))

# ---- RAG-like Ticket Retrieval with Nearest Neighbors ----

full_vec = vectorizer.fit_transform(df['issue'])
retriever = NearestNeighbors(n_neighbors=5, metric='cosine')
retriever.fit(full_vec)

def retrieve_similar_tickets(issue_text, top_k=5):
    query_vec = vectorizer.transform([issue_text])
    distances, indices = retriever.kneighbors(query_vec, n_neighbors=top_k)
    return df.iloc[indices[0]][['ticketId', 'issue', 'nextAction']].to_dict(orient='records')


# ---- Full Triage Function ----

def triage_ticket(issue_text):
    vec = vectorizer.transform([issue_text])
    urgency_score = clf.predict(vec)[0]
    similar_tickets = retrieve_similar_tickets(issue_text)

    return {
        "urgencyScore": int(urgency_score),
        "suggestedNextAction": "⚠️ Placeholder: fine-tune LLM or use rules",
        "similarHistoricalTickets": similar_tickets
    }

# ---- Example Test Run ----

example_issue = "Shipment of critical lab instruments denied customs entry due to missing import permits. Customer needs for testing."
triage_result = triage_ticket(example_issue)

print("\n=== Triage Result ===")
print(json.dumps(triage_result, indent=2))
