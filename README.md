# Data Analysis Agent

A CSV analysis app built with Python, Streamlit, pandas, and LangGraph. It helps users inspect a dataset, review cleaning results, and ask questions about the data in plain language.

## Features

- Upload a CSV file
- Run a LangGraph workflow to inspect and clean the dataset
- View the cleaned dataset and download it as a CSV
- Review cleaning logs and analysis results
- Ask questions about columns, missing values, distinct values, and numeric statistics
- Keep conversation context for supported follow-up questions

## Tech Stack

- Python
- Streamlit
- pandas
- LangGraph

## Project Structure

```text
Data_Analysis_Agent/
├── agents/
│   ├── analyst.py
│   ├── cleaner.py
│   ├── cleaning_planner.py
│   ├── reporter.py
│   ├── validator.py
│   └── workflow.py
├── reports/
├── app.py
├── main.py
├── requirements.txt
└── .gitignore
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/ahmedbaig123/https://github.com/ahmedbaig123/Data_analysis_agent.git
   cd Data_analysis_agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the app in your browser.
2. Upload a CSV file.
3. Review the cleaned dataset and workflow report.
4. Ask questions in the **Ask your data** tab.
5. Download the cleaned CSV if needed.

## Current Scope

The question-answering feature uses pandas calculations and supports a defined set of dataset questions. It is not a general-purpose chatbot, and it may not understand every natural-language question or follow-up.

## Future Improvements

- Expand the range of supported questions and follow-ups
- Add automated tests and evaluation
- Improve deployment and error handling

## License

No license has been added yet.
