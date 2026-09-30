# FinMate: Agentic Equity-Research Assistant

Answers questions about Indian listed companies only from verified summaries, and uses live tools
for share prices, currency conversion and date. Information only, no investment advice.

## Run
1. `pip install -r requirements.txt`
2. Get a free key from console.groq.com and set it: `export GROQ_API_KEY=your_key` (never commit it)
3. Write 12 company files in `data/` from public annual reports (copy `data/_TEMPLATE.md`), e.g. `TCS.md`
4. `python ingest.py`
5. `streamlit run app.py`
6. Testing: `python tests.py` (10 test cases). Evaluation: write `qa.csv` (20 rows: question,expected_company) then `python evaluate.py` (uses the included `tool_qa.csv`)

## Files
agent.py (LangGraph pipeline) · ingest.py · app.py · tests.py · evaluate.py
