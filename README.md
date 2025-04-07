## Environment Requirements

- Python 3.9.2  
---

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## API Key Configuration

To run the main program, you need to provide an OpenAI API key.

1. Create a `.env` file in the root directory.  
2. Add the following line:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Running the Code

Navigate to the `experiment_llm_boundry/` directory and run the main script:

```bash
cd experiment_llm_boundry
python main.py
```

- The prompts and experiment parameters can be configured in the `config.json` file.

---