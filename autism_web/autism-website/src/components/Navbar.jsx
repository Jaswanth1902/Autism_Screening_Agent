import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "./Navbar.css";
import logo from "../assets/images/logo.jpg";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const scrollToSection = (id) => {
    // If we're on the home page, scroll to section
    if (location.pathname === "/") {
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "start" });
        setMenuOpen(false);
      }
    } else {
      // If we're on another page, navigate to home first
      navigate("/");
      // Wait for navigation, then scroll
      setTimeout(() => {
        const element = document.getElementById(id);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 100);
      setMenuOpen(false);
    }
  };

  return (
    <nav className="navbar">
      <div className="nav-container">

        {/* Logo */}
        <div 
          className="nav-logo"
          onClick={() => scrollToSection("home")}
        >
          <img src={logo} alt="Autism AI Logo" className="logo-img" />
          <span className="logo-text">Autism AI</span>
        </div>

        {/* Hamburger Icon */}
        <div
          className={`menu-icon ${menuOpen ? "active" : ""}`}
          onClick={() => setMenuOpen(!menuOpen)}
        >
          <div className="bar"></div>
          <div className="bar"></div>
          <div className="bar"></div>
        </div>

        {/* Menu Links */}
        <ul className={`nav-links ${menuOpen ? "open" : ""}`}>
          <li>
            <button onClick={() => scrollToSection("home")}>Home</button>
          </li>
          <li>
            <button onClick={() => scrollToSection("about")}>About Autism</button>
          </li>
          <li>
            <button onClick={() => scrollToSection("models")}>Model Details</button>
          </li>
          <li>
            <button onClick={() => scrollToSection("empowerment")}>
              Empowerment
            </button>
          </li>
          <li>
            <button onClick={() => scrollToSection("contact")}>Contact</button>
          </li>
        </ul>

      </div>
    </nav>
  );
};

export default Navbar;
