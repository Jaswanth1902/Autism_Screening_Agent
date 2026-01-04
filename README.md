# Autism Screening Agent 💙

A sophisticated, AI-powered web application for early autism screening in children. This agentic tool combines a React-based interactive frontend with a Flask-served machine learning model to provide real-time, compassionate, and actionable screening results.

## Features

-   **Interactive Screening**: A 10-question behavioral screening form designed for parents and caregivers.
-   **AI Prediction**: Uses a pre-trained ML model (served via Flask) to analyze inputs and detect potential ASD traits.
-   **3-Tier Result System**:
    -   🟢 **Low Risk**: Reassuring feedback for low probability scores.
    -   🟡 **Moderate Risk**: Cautionary advice for borderline cases.
    -   🟠 **High Risk**: Clear recommendations for professional evaluation when traits are detected.
-   **Resources & Support**: Provides contact information for local autism centers and specialists in Karnataka.
-   **Private & Secure**: Runs locally on your machine.

## Tech Stack

-   **Frontend**: React.js, CSS3
-   **Backend**: Python, Flask
-   **ML**: Scikit-learn, XGBoost/LightGBM (Stacking Classifier)

## Setup & Run

1.  **Backend**:
    ```bash
    pip install -r requirements.txt
    python app.py
    ```

2.  **Frontend**:
    ```bash
    cd autism_web/autism-website
    npm install
    npm start
    ```

3.  Access the app at `http://localhost:3000`.

## License

[MIT](LICENSE)
