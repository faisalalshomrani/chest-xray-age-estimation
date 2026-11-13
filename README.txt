# Chest X-ray Age Estimation — Streamlit Prototype

## Files
- `app.py`: Streamlit interface and inference code.
- `requirements.txt`: Python dependencies.
- `densenet121_age_model.pth`: **Add your exported model file here.**

## 1. Put the model in this folder
Copy `densenet121_age_model.pth` into the same folder as `app.py`.

The folder should look like this:

    Chest_Xray_Age_Prototype/
        app.py
        requirements.txt
        README.txt
        densenet121_age_model.pth

## 2. Open a terminal in this folder

On Windows, open the folder in File Explorer, click the address bar, type `cmd`, and press Enter.

## 3. Create and activate a virtual environment (recommended)

    py -m venv .venv
    .venv\Scripts\activate

If `py` is not recognised, try `python` instead.

## 4. Install packages

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

PyTorch installation can vary by operating system and hardware. If installation fails, use the official PyTorch installation selector:
https://pytorch.org/get-started/locally/

## 5. Run Streamlit

    python -m streamlit run app.py

The app should open in your browser, usually at:
http://localhost:8501

## Important model details
The application recreates DenseNet121 with a single regression output and loads the state dictionary exported from the trained baseline checkpoint. It applies the same 224 x 224 resize and ImageNet normalisation described in the training workflow.

The reported validation/test MAE values are cohort-level metrics and do not represent an individual prediction's confidence interval.

## Research-use notice
This is a research prototype, not a medical device. Do not use its predictions for diagnosis, treatment, or patient management.
