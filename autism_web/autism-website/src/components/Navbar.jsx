import React, { useState } from "react";
import "./Navbar.css";
import logo from "../assets/images/logo.jpg"; // add a small autism awareness logo

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="nav-container">
        {/* Logo */}
        <div className="nav-logo">
          <img src={logo} alt="Autism AI Logo" className="logo-img" />
          <span className="logo-text">Autism AI</span>
        </div>

        {/* Hamburger Icon (mobile) */}
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
          <li><a href="/">Home</a></li>
          <li><a href="/about">About Autism</a></li>
          <li><a href="/models">Detection Models</a></li>
          <li><a href="/empowerment">Empowerment</a></li>
          <li><a href="/contact">Contact</a></li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
