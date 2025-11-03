import React from "react";
import "./Footer.css";
import { FaLinkedin, FaGithub, FaEnvelope } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-container">
        {/* About */}
        <div className="footer-about">
          <h3>Autism AI Project 💙</h3>
          <p>
            Empowering awareness and early detection of Autism using modern AI models.
          </p>
        </div>

        {/* Contact */}
        <div className="footer-contact">
          <h4>Contact Me</h4>
          <p>Kannali Chenchu Deepthi</p>
          <p>Email: deepthireddy2363@gmail.com</p>
        </div>

        {/* Socials */}
        <div className="footer-socials">
          <a href="https://linkedin.com/in/deepthi-reddy-630a72240" target="_blank" rel="noreferrer">
            <FaLinkedin />
          </a>
          <a href="https://github.com/deepthi2363" target="_blank" rel="noreferrer">
            <FaGithub />
          </a>
          <a href="mailto:deepthireddy2363@gmail.com">
            <FaEnvelope />
          </a>
        </div>
      </div>

      <div className="footer-bottom">
        © {new Date().getFullYear()} Deepthi | All Rights Reserved 🌍
      </div>
    </footer>
  );
};

export default Footer;
