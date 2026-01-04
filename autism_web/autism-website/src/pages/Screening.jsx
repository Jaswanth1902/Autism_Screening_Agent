import React, { useState } from "react";
import "./Screening.css";

export default function Screening() {
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [form, setForm] = useState({
    age: "",
    gender: "",
    jaundice: "0",
    autism_in_family: "0",
    A1_Score: "",
    A2_Score: "",
    A3_Score: "",
    A4_Score: "",
    A5_Score: "",
    A6_Score: "",
    A7_Score: "",
    A8_Score: "",
    A9_Score: "",
    A10_Score: "",
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Use 127.0.0.1 to avoid localhost IPv4/IPv6 resolution issues
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Submission error:", err);
      setError(err.message || "Failed to get prediction. Ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const nextStep = () => setStep((s) => s + 1);
  const prevStep = () => setStep((s) => s - 1);

  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <>
            <h2 className="section-label">General Information</h2>
            <div className="form-group">
              <label>Age</label>
              <input
                type="number"
                name="age"
                placeholder="Enter age"
                value={form.age}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label>Gender</label>
              <select name="gender" value={form.gender} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            <div className="form-group">
              <label>Jaundice at birth?</label>
              <select name="jaundice" value={form.jaundice} onChange={handleChange}>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label>Family history of autism?</label>
              <select
                name="autism_in_family"
                value={form.autism_in_family}
                onChange={handleChange}
              >
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
          </>
        );
      case 1:
        return (
          <>
            <h2 className="section-label">Behavioral Questions (1/5)</h2>
            <div className="form-group">
              <label>Q1: Does your child look at you when you call their name?</label>
              <select name="A1_Score" value={form.A1_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Q2: Is it easy to get eye contact with your child?</label>
              <select name="A2_Score" value={form.A2_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
          </>
        );
      case 2:
        return (
          <>
            <h2 className="section-label">Behavioral Questions (2/5)</h2>
            <div className="form-group">
              <label>Q3: Does your child point to indicate interest?</label>
              <select name="A3_Score" value={form.A3_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Q4: Does your child pretend play (e.g., toy phone)?</label>
              <select name="A4_Score" value={form.A4_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
          </>
        );
      case 3:
        return (
          <>
            <h2 className="section-label">Behavioral Questions (3/5)</h2>
            <div className="form-group">
              <label>Q5: Does your child enjoy playing with other children?</label>
              <select name="A5_Score" value={form.A5_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Q6: Does your child respond to emotions of others?</label>
              <select name="A6_Score" value={form.A6_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
          </>
        );
      case 4:
        return (
          <>
            <h2 className="section-label">Behavioral Questions (4/5)</h2>
            <div className="form-group">
              <label>Q7: Does your child use gestures (e.g., waving)?</label>
              <select name="A7_Score" value={form.A7_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Q8: Does your child show repetitive movements?</label>
              <select name="A8_Score" value={form.A8_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
          </>
        );
      case 5:
        return (
          <>
            <h2 className="section-label">Behavioral Questions (5/5)</h2>
            <div className="form-group">
              <label>Q9: Does your child get upset by small changes in routine?</label>
              <select name="A9_Score" value={form.A9_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
            <div className="form-group">
              <label>Q10: Does your child have unusually intense interests?</label>
              <select name="A10_Score" value={form.A10_Score} onChange={handleChange} required>
                <option value="">Select</option>
                <option value="1">Yes</option>
                <option value="0">No</option>
                <option value="0.5">Not Sure</option>
              </select>
            </div>
          </>
        );
      default:
        return null;
    }
  };

  if (result) {
    const prob = result.probability;
    let statusClass = "low";
    let statusLabel = "Low Screening Score 🌟";
    let statusMessage = "Our AI model did not detect significant indicators of Autism. Continue monitoring developmental milestones.";
    let barColor = "linear-gradient(90deg, #4ade80, #22c55e)"; // Green

    if (prob >= 0.35 && prob < 0.7) {
      statusClass = "moderate";
      statusLabel = "Moderate / Borderline ⚠️";
      statusMessage = "The screening suggests some behavioral patterns that may warrant attention. We recommend a closer observation or a consultation.";
      barColor = "linear-gradient(90deg, #facc15, #eab308)"; // Yellow
    } else if (prob >= 0.7) {
      statusClass = "high";
      statusLabel = "Significant Traits Detected 🧬";
      statusMessage = "Our AI model has identified behavioral patterns often associated with Autism Spectrum Disorder. A professional evaluation is highly recommended.";
      barColor = "linear-gradient(90deg, #fb923c, #ea580c)"; // Orange
    }

    return (
      <div className="screening-container">
        <div className="result-card">
          <h2 className="result-title">Screening Result 📋</h2>
          <div className={`status-badge ${statusClass}`}>
            {statusLabel}
          </div>
          
          <div className="probability-section">
            <p>Assessment Depth Indicator</p>
            <div className="progress-bar-bg">
              <div 
                className="progress-bar-fill" 
                style={{ 
                  width: `${Math.max(prob * 100, 5)}%`, 
                  background: barColor
                }}
              ></div>
            </div>
            <span className="prob-text">AI Pattern Analysis Confidence</span>
          </div>

          <div className="result-info">
            <p>{statusMessage}</p>
          </div>

          <div className="next-steps-container">
            <h3>Suggested Next Steps ✨</h3>
            <ul className="suggestions-list">
              <li>📅 Schedule a peaceful developmental checkup with your pediatrician.</li>
              <li>memo Keep a gentle log of behavioral patterns to share with specialists.</li>
              <li>💙 Explore local community support groups for neurodiversity.</li>
              <li>🏠 Create a structured yet flexible daily routine at home.</li>
            </ul>
          </div>

          <div className="resources-container">
            <h3>Local Support & Specialists 📍</h3>
            <div className="resource-grid">
              <div className="resource-item">
                <strong>Academy for Severe Handicaps and Autism (ASHA)</strong>
                <p>📞 +91 80 2322 5279</p>
                <p>📍 Basaveshwaranagar, Bangalore, Karnataka</p>
              </div>
              <div className="resource-item">
                <strong>Com DEALL Trust</strong>
                <p>📞 +91 80 2580 0826</p>
                <p>📍 Cox Town, Bangalore, Karnataka</p>
              </div>
              <div className="resource-item">
                <strong>Bubbles Centre for Autism</strong>
                <p>📞 +91 80 4091 8971</p>
                <p>📍 Yelahanka, Bangalore, Karnataka</p>
              </div>
              <div className="resource-item">
                <strong>Parijma Neurodiagnostic & Rehab Centre</strong>
                <p>📞 +91 80 2344 5555</p>
                <p>📍 Wilson Garden, Bangalore, Karnataka</p>
              </div>
            </div>
          </div>

          <button className="reset-btn" onClick={() => {setResult(null); setStep(0);}}>
            Re-take Screening
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="screening-container">
      <div className="screening-card">
        <h1 className="screening-title">Autism Screening Test 💙</h1>
        {!loading && <div className="step-indicator">Step {step + 1} of 6</div>}
        
        {error && <div className="error-msg">⚠️ {error}</div>}
        
        {loading ? (
          <div className="loading-container">
            <div className="spinner"></div>
            <p>Analyzing behavioral patterns...</p>
          </div>
        ) : (
          <>
            <p className="screening-sub">
              Please fill out the following information for initial AI-based autism screening.
            </p>

            <form onSubmit={handleSubmit} className="screening-form">
              {renderStep()}

              <div className="nav-buttons">
                {step > 0 && (
                  <button type="button" className="prev-btn" onClick={prevStep}>
                    Previous
                  </button>
                )}
                {step < 5 ? (
                  <button type="button" className="next-btn" onClick={nextStep}>
                    Next
                  </button>
                ) : (
                  <button className="submit-btn" type="submit">
                    Submit Screening
                  </button>
                )}
              </div>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
