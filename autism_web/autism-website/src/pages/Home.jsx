import React from "react";
import "./Home.css";
import heroImg from "../assets/images/hero.jpg";
import awarenessImg from "../assets/images/awareness.jpg";
import detectImg from "../assets/images/detect.jpeg";
import featureImg from "../assets/images/features.png";
import datasetImg from "../assets/images/dataset.png";
import usersImg from "../assets/images/users.png";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const navigate = useNavigate();

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="home-page" id="home">

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1 className="hero-title">Early Autism Detection Using AI</h1>
          <p className="hero-desc">
            Smart screening for early behavioral indicators to support families and doctors.
          </p>

          <button className="hero-btn" onClick={() => scrollTo("about")}>
            Learn More
          </button>
        </div>
        <img src={heroImg} alt="Autism Awareness" className="hero-img" />
      </section>

      {/* About Autism */}
      <section id="about" className="section">
        <img src={awarenessImg} alt="About Autism" className="section-img" />
        <div>
          <h2>What is Autism?</h2>
          <p>
            Autism Spectrum Disorder (ASD) affects social interaction, communication,
            and sensory processing. Early intervention helps young minds adapt,
            learn and thrive in their unique ways.
          </p>
          <p>
            Recognizing early signs makes a big difference in emotional and
            learning outcomes throughout life.
          </p>
        </div>
      </section>

      {/* AI Models */}
      <section id="models" className="section reverse">
        <img src={detectImg} alt="AI Detection" className="section-img" />
        <div>
          <h2>How Our AI Helps</h2>
          <p>
            We use advanced ML models like CatBoost, XGBoost, LightGBM and Random Forest
            to detect behavioral patterns with high accuracy.
          </p>
          <p>
            Using SHAP explainability, predictions are transparent so caregivers
            understand how AI arrives at results.
          </p>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="section">
        <div className="text-block">
          <h2>Why Our System Stands Out</h2>
          <ul className="feature-list">
            <li>Explainable AI (SHAP insights)</li>
            <li>High prediction accuracy</li>
            <li>Fast and secure screening</li>
            <li>Accessible and easy-to-use interface</li>
          </ul>
        </div>
        <img src={featureImg} alt="Features" className="section-img" />
      </section>

      {/* Users */}
      <section id="users" className="section reverse">
        <img src={usersImg} alt="Users" className="section-img" />
        <div>
          <h2>Who Can Use This?</h2>
          <ul className="details-list">
            <li>Parents noticing early behavioral differences</li>
            <li>Healthcare professionals supporting evaluation</li>
            <li>Educators monitoring social learning patterns</li>
          </ul>
          <p>We assist decisions, not replace them.</p>
        </div>
      </section>

      {/* Dataset */}
      <section id="dataset" className="section">
        <img src={datasetImg} alt="Dataset" className="section-img" />
        <div>
          <h2>Research & Performance</h2>
          <ul className="details-list">
            <li>Questioning Standard: ISAA</li>
            <li>Accuracy: 94.2%</li>
            <li>AUC Score: 0.96</li>
            <li>Stable performance across age groups</li>
          </ul>
          <p>
            Dataset used: Validated ASD screening dataset with medically reviewed features.
          </p>
        </div>
      </section>

      {/* Responsible Use */}
      <section className="responsible">
        <p>
          We support awareness and screening. Clinical professionals must confirm diagnosis.
        </p>
      </section>

      {/* CTA */}
      <section id="empowerment" className="cta-section">
        <h2>Empowering Early Diagnosis 💙</h2>
        <p>Try our model and help build a more aware society.</p>
        <button className="cta-btn" onClick={() => navigate("/screening")}>
          Get Started
        </button>
      </section>

    </div>
  );
};

export default Home;
