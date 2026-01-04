import React from "react";
import "./Footer.css";
import { FaLinkedin, FaGithub, FaEnvelope } from "react-icons/fa";

const Footer = () => {
  return (
    <footer className="footer" id="contact">
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
          <p>K. Sai Jaswanth Reddy</p>
          <p>Email: ksaijaswanthr.cs24@gmail.com</p>
        </div>

        {/* Socials */}
        <div className="footer-socials">
          <a href="#!" onClick={(e) => e.preventDefault()}>
            <FaLinkedin />
          </a>
          <a href="#!" onClick={(e) => e.preventDefault()}>
            <FaGithub />
          </a>
          <a href="mailto:ksaijaswanthr.cs24@gmail.com">
            <FaEnvelope />
          </a>
        </div>
      </div>

      <div className="footer-bottom">
        © {new Date().getFullYear()} K. Sai Jaswanth Reddy | All Rights Reserved 🌍
      </div>
    </footer>
  );
};

export default Footer;
