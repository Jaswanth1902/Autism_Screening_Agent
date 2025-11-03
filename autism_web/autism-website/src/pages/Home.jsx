import React from "react";
import "./Home.css";
import heroImg from "../assets/images/hero.jpg";
import awarenessImg from "../assets/images/awareness.jpg";
import detectImg from "../assets/images/detect.jpeg";

const Home = () => {
  return (
    <div className="home-page">
      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">
            Early Detection of Autism Using AI 🤖
          </h1>
          <p className="hero-desc">
            Our intelligent model helps identify autism spectrum patterns early
            — supporting faster intervention, better understanding, and
            awareness for families and doctors.
          </p>
          <button className="hero-btn">Learn More</button>
        </div>
        <img src={heroImg} alt="Autism Awareness" className="hero-img" />
      </section>

      {/* About Autism */}
      <section className="about-section">
        <img src={awarenessImg} alt="About Autism" />
        <div className="about-content">
          <h2 className="about-title">What is Autism?</h2>
          <p className="about-text">
            Autism Spectrum Disorder (ASD) affects how people communicate,
            interact, and process information. It appears in early childhood and
            continues throughout life. Awareness and timely detection play a
            vital role in improving social and learning outcomes.
          </p>
        </div>
      </section>

      {/* Detection Info */}
      <section className="detect-section">
        <img src={detectImg} alt="AI Detection" />
        <div className="detect-content">
          <h2 className="detect-title">How Our AI Model Helps</h2>
          <p className="detect-text">
            Our machine learning pipeline analyzes behavioral and medical
            indicators to detect potential autism patterns. It combines
            XGBoost, CatBoost, LightGBM, and Random Forests — providing
            explainable results with SHAP analysis and probability calibration.
          </p>
        </div>
      </section>

      {/* Call to Action */}
      <section className="cta-section">
        <h2 className="cta-title">
          Empowering Awareness & Early Diagnosis 💙
        </h2>
        <p className="cta-text">
          Join us in spreading awareness and improving access to early autism
          screening. Explore our research, try our models, and make a
          difference.
        </p>
        <button className="cta-btn">Get Started</button>
      </section>
    </div>
  );
};

export default Home;
