import React from "react";
import heroImg from "../assets/images/hero.jpg"; // Add an image under src/assets/

const HeroSection = () => {
  return (
    <section className="flex flex-col md:flex-row items-center justify-between px-10 py-16 bg-blue-50">
      <div className="max-w-xl">
        <h2 className="text-4xl font-bold text-blue-700 mb-4">
          Understanding Autism through Technology
        </h2>
        <p className="text-gray-700 mb-6">
          Our AI-powered Autism Detection Project helps in identifying early
          signs of Autism Spectrum Disorder (ASD). Empowering families,
          healthcare professionals, and educators with data-driven insights.
        </p>
        <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-5 rounded-lg transition">
          Learn More
        </button>
      </div>
      <img
        src={heroImg}
        alt="Autism Awareness"
        className="w-full md:w-1/2 rounded-xl mt-8 md:mt-0 shadow-lg"
      />
    </section>
  );
};

export default HeroSection;
