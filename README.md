# AutoDoc AI — Automotive HLD Document Analysis Assistant

**Capstone prototype | Applied AI/ML | Python + Streamlit + TF-IDF retrieval + optional Ollama local LLM**

## What this prototype does
- Upload an engineering PDF and extract text with page numbers.
- Split extracted text into overlapping chunks and rank them against a question with TF-IDF/cosine similarity.
- Show evidence passages and PDF page citations.
- Answer using a local Ollama model when available; otherwise use an extractive, evidence-only fallback.
- Run a small, transparent retrieval evaluation against the included synthetic HLD and question set.

This project follows the direction of the supplied *Automotive Engineering AI: Project Case Study Document*, Case Study 1 (AUTOSAR HLD Document Analysis Assistant): document ingestion, searchable architecture knowledge, cited answers, and human review. The supplied case study is a proposal/reference document, **not** an actual AUTOSAR HLD dataset. The included example HLD is synthetic and illustrative.

## Quick start (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP.
3. Open Command Prompt in this folder and run:
   ```bat
   setup.bat
   ```
4. Activate the environment if setup did not keep it active:
   ```bat
   .venv\Scripts\activate
   ```
5. Run:
   ```bat
   streamlit run Code\app.py
   ```
6. In the app, upload `Input_Data\synthetic_autosar_hld.txt` (or an authorized HLD PDF). Text files are supported too.

## Optional local LLM (Ollama)
The app does not require an API key. If Ollama is installed and running locally:
1. Pull a model suitable for your hardware, for example `ollama pull llama3.2:3b`.
2. Ensure `ollama serve` is running (some installations start it automatically).
3. In the app sidebar, enable Ollama and select the installed model name.
4. If Ollama is unavailable, the app falls back to an extractive answer and labels it clearly.

Model use, exact model version, and observed availability must be recorded accurately in `Model_Prompts_Config/model_config.json`. Do not claim the local LLM was used unless it was actually enabled and a response succeeded.

## Project files
- `Code/app.py` — Streamlit UI, ingestion, retrieval, citations, optional Ollama generation.
- `Code/retrieval.py` — reusable retrieval functions.
- `Code/evaluate.py` — deterministic retrieval evaluation; outputs JSON and CSV.
- `Input_Data/synthetic_autosar_hld.txt` — synthetic demo knowledge base, not production AUTOSAR content.
- `Input_Data/evaluation_questions.csv` — predefined questions and expected evidence terms.
- `Model_Prompts_Config/system_prompt.txt` — grounding prompt.
- `Model_Prompts_Config/model_config.json` — configuration template.
- `Evaluation_Results/` — place actual run outputs here.
- `Documentation/Technical_Report.pdf` — report template describing the prototype and its limitations.
- `Documentation/Demo_Script.md` — 5–10 minute recording plan.
- `Synopsis/Project_Synopsis_Template.md` and `Declarations/Student_Declaration_Template.md` — complete and get required faculty/student signatures.

## Important
- The demo dataset is synthetic; do not present it as an OEM/customer document or certified AUTOSAR standard.
- The evaluation script measures retrieval evidence matching, not answer correctness, engineering safety, or AUTOSAR compliance.
- Run `python Code\evaluate.py` to produce an actual local evaluation result before reporting metrics.
- Replace placeholders with your own name, PRN, class, faculty details, and approval evidence.
- The notice's stated deadline (8 October 2026, 6:00 PM) has passed as of 15 October 2026. Confirm whether the placement cell will accept late submissions.
