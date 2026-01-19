import React, { useState, useRef } from "react";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import "./Screening.css";

const TRANSLATIONS = {
  en: {
    title: "Autism Screening Assessment 💙",
    next: "Next",
    back: "Back",
    submit: "Complete Assessment",
    generalInfo: "General Information",
    age: "Age",
    gender: "Gender",
    jaundice: "Jaundice at birth?",
    familyHistory: "Family history of autism?",
    rarely: "Rarely / Never",
    sometimes: "Sometimes",
    always: "Frequently / Always",
    lowRisk: "Low Risk 🌟",
    moderateRisk: "Moderate Risk ⚠️",
    highRisk: "High Risk Detected 🧬",
    resultTitle: "Autism Screening Analysis 📋",
    scoreLabel: "Assessment Score",
    recommendation: "Recommended Actions ✨",
    resources: "Karnataka Autism Resources 📍",
    startOver: "New Screening",
    downloadPDF: "Download Clinical Report 📄",
    select: "Select",
  },
  kn: {
    title: "ಆಟಿಸಂ ಸ್ಕ್ರೀನಿಂಗ್ ಮೌಲ್ಯಮಾಪನ 💙",
    next: "ಮುಂದೆ",
    back: "ಹಿಂದೆ",
    submit: "ಮೌಲ್ಯಮಾಪನ ಪೂರ್ಣಗೊಳಿಸಿ",
    generalInfo: "ಸಾಮಾನ್ಯ ಮಾಹಿತಿ",
    age: "ವಯಸ್ಸು",
    gender: "ಲಿಂಗ",
    jaundice: "ಜನ್ಮ ಸಮಯದಲ್ಲಿ ಕಾಮಾಲೆ ಇತ್ತೆ?",
    familyHistory: "ಕುಟುಂಬದಲ್ಲಿ ಆಟಿಸಂ ಇತಿಹಾಸವಿದೆಯೇ?",
    rarely: "ಅಪರೂಪವಾಗಿ / ಎಂದಿಗೂ ಇಲ್ಲ",
    sometimes: "ಕೆಲವೊಮ್ಮೆ",
    always: "ಪದೇ ಪದೇ / ಯಾವಾಗಲೂ",
    lowRisk: "ಕಡಿಮೆ ಅಪಾಯ 🌟",
    moderateRisk: "ಮಧ್ಯಮ ಅಪಾಯ ⚠️",
    highRisk: "ಹೆಚ್ಚಿನ ಅಪಾಯ ಪತ್ತೆಯಾಗಿದೆ 🧬",
    resultTitle: "ಆಟಿಸಂ ಸ್ಕ್ರೀನಿಂಗ್ ವಿಶ್ಲೇಷಣೆ 📋",
    scoreLabel: "ಮೌಲ್ಯಮಾಪನ ಅಂಕ",
    recommendation: "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ರಮಗಳು ✨",
    resources: "ಕರ್ನಾಟಕ ಆಟಿಸಂ ಸಂಪನ್ಮೂಲಗಳು 📍",
    startOver: "ಹೊಸ ಸ್ಕ್ರೀನಿಂಗ್",
    downloadPDF: "ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ 📄",
    select: "ಆಯ್ಕೆ ಮಾಡಿ",
  },
  hi: {
    title: "ऑटिज़्म स्क्रीनिंग मूल्यांकन 💙",
    next: "आगे",
    back: "पीछे",
    submit: "मूल्यांकन पूरा करें",
    generalInfo: "सामान्य जानकारी",
    age: "आयु",
    gender: "लिंग",
    jaundice: "जन्म के समय पीलिया?",
    familyHistory: "ऑटिज़्म का पारिवारिक इतिहास?",
    rarely: "शायद ही कभी / कभी नहीं",
    sometimes: "कभी-कभी",
    always: "अक्सर / हमेशा",
    lowRisk: "कम जोखिम 🌟",
    moderateRisk: "मध्यम जोखिम ⚠️",
    highRisk: "उच्च जोखिम पाया गया 🧬",
    resultTitle: "ऑटिज़्म स्क्रीनिंग विश्लेषण 📋",
    scoreLabel: "मूल्यांकन स्कोर",
    recommendation: "अनुशंसित कार्रवाई ✨",
    resources: "कर्नाटक ऑटिज़्म संसाधन 📍",
    startOver: "नई स्क्रीनिंग",
    downloadPDF: "रिपोर्ट डाउनलोड करें 📄",
    select: "चुनें",
  }
};

const DOMAINS = [
  {
    id: "social",
    title: "Social Relationship & Reciprocity",
    gates: [
      { id: "Q1", text: "Does your child look at you when you call his/her name?", isInverted: true, tip: "Does the child respond to their name by making eye contact?" },
      { id: "Q2", text: "Is it easy for you to get eye contact with your child?", isInverted: true, tip: "Does the child naturally look at your face during interaction?" },
    ],
    details: [
      { id: "Q3", text: "Does your child point to indicate interest (e.g., in a toy)?", isInverted: true, tip: "Does the child use their finger to show you something they like?" },
      { id: "Q4", text: "Does your child pretend play (e.g., talk on a toy phone)?", isInverted: true, tip: "Does the child engage in imaginative play like feeding a doll or driving a car?" },
      { id: "Q5", text: "Does your child enjoy playing with other children?", isInverted: true, tip: "Does the child show interest in peers and join in their games?" },
      { id: "Q6", text: "Does your child respond to emotions of others (e.g., comfort someone)?", isInverted: true, tip: "Does the child notice when someone is sad or happy and react?" },
      { id: "Q7", text: "Does your child use gestures other than pointing (e.g., waving goodbye)?", isInverted: true, tip: "Using hands to signal 'hello', 'bye', or 'come here'." },
      { id: "Q8", text: "Does your child show repetitive movements (e.g., hand-flapping)?", tip: "Repeated actions like rocking, spinning, or finger flicking." },
      { id: "Q9", text: "Does your child get upset by small changes in routine?", tip: "Does the child have difficulty adjusting to a new route or a change in plans?" },
    ]
  },
  {
    id: "emotional",
    title: "Emotional & Behavioral Response",
    gates: [
      { id: "Q10", text: "Does your child have unusually intense interests?", tip: "Being extremely focused on specific topics like trains, maps, or fans." },
    ],
    details: [
      { id: "Q11", text: "Does your child show very strong or exaggerated emotional reactions?", tip: "Extreme crying or laughter that seems out of proportion to the event." },
      { id: "Q12", text: "Does your child have emotional outbursts that seem to be just for self-comfort?", tip: "Expressed emotions that don't seem linked to external events." },
      { id: "Q13", text: "Does your child seem unaware of common dangers (like heights or traffic)?", tip: "Lack of typical fear in situations that could be risky." },
      { id: "Q14", text: "Does your child get very excited or jump/flap hands for no clear reason?", tip: "Unexpected physical displays of excitement." },
    ]
  },
  {
    id: "communication",
    title: "Speech-Language & Communication",
    gates: [
      { id: "Q15", text: "Did your child ever learn to say words and then stop using them?", tip: "Losing speech or social skills they previously had." },
      { id: "Q20", text: "Is it difficult for your child to keep a back-and-forth conversation going?", tip: "Trouble taking turns in a verbal exchange." },
    ],
    details: [
      { id: "Q16", text: "Does your child have difficulty using gestures like waving or nodding?", tip: "Not using typical body language to communicate." },
      { id: "Q17", text: "Does your child repeat the same phrases or words over and over?", tip: "Saying things repeatedly even if they don't fit the context." },
      { id: "Q18", text: "Does your child repeat exactly what you just said (echoing)?", tip: "Repeating your question instead of answering it." },
      { id: "Q19", text: "Does your child make unusual squealing or high-pitched noises?", tip: "Sudden vocal sounds that aren't words." },
      { id: "Q21", text: "Does your child babble or use 'jargon' that doesn't sound like real words?", tip: "Making sounds that mimic conversation but have no clear meaning." },
      { id: "Q22", text: "Does your child mix up pronouns (e.g., saying 'You want' when they mean 'I')?", tip: "Referring to themselves as 'You' or by their name." },
      { id: "Q23", text: "Does your child struggle to understand the hidden or implied meaning in what people say?", tip: "Taking everything literally (e.g., missing sarcasm or jokes)." },
    ]
  },
  {
    id: "behavior",
    title: "Behavior Patterns",
    gates: [
      { id: "Q24", text: "Does your child rock their body, flick fingers, or flap hands repeatedly?", tip: "Repetitive physical habits that occupy a lot of time." },
    ],
    details: [
      { id: "Q25", text: "Does your child seem unusually attached to specific objects like strings or toy wheels?", tip: "Focusing on parts of objects or non-toy items intensely." },
      { id: "Q26", text: "Is your child constantly on the go and unable to sit still?", tip: "High levels of physical activity compared to peers." },
      { id: "Q27", text: "Does your child ever hit, bite, or push others when frustrated?", tip: "Aggressive behavior during social interactions." },
      { id: "Q28", text: "Does your child have intense meltdowns or temper tantrums?", tip: "Severe reactions to frustration or sensory overload." },
      { id: "Q29", text: "Does your child ever hurt themselves, such as by head banging or biting?", tip: "Self-injurious actions when upset." },
      { id: "Q30", text: "Does your child insist on things staying the same at all times?", tip: "Upset by changes in their environment or routine." },
    ]
  },
  {
    id: "sensory",
    title: "Sensory Aspects",
    gates: [
      { id: "Q31", text: "Does your child seem very bothered by loud noises, bright lights, or certain textures?", tip: "Over-sensitivity to the environment." },
    ],
    details: [
      { id: "Q32", text: "Does your child frequently 'zone out' or stare into space for long periods?", tip: "Appearing to be in their own world." },
      { id: "Q33", text: "Does your child have trouble smoothly following a moving toy with their eyes?", tip: "Difficulty tracking objects as they move." },
      { id: "Q34", text: "Does your child look at things from unusual angles (like the corner of their eye)?", tip: "Non-typical ways of visually inspecting items." },
      { id: "Q35", text: "Does your child seem not to feel pain from bumps or falls that usually hurt?", tip: "Under-sensitivity to physical pain." },
      { id: "Q36", text: "Does your child check objects by smelling, licking, or touching them in unusual ways?", tip: "Exploring the world through senses in non-typical ways." },
    ]
  },
  {
    id: "cognitive",
    title: "Cognitive Component",
    gates: [
      { id: "Q37", text: "Is your child's attention often inconsistent or easily distracted?", tip: "Difficulty focusing on one task or person." },
    ],
    details: [
      { id: "Q38", text: "Does it often take your child several seconds to react when you call them?", tip: "Delayed response to social approach." },
      { id: "Q39", text: "Does your child have an unusual memory for patterns or facts, but forgets names?", tip: "Exceptional memory for non-social information." },
      { id: "Q40", text: "Does your child have an exceptional talent in a specific area like math or music?", tip: "Showing 'savant' skills in specific domains." },
    ]
  }
];

const KARNATAKA_CENTERS = [
  { name: "ASHA (Bangalore)", phone: "+91 80 2322 5279", area: "Basaveshwaranagar", type: "Academy for Autism" },
  { name: "Com DEALL (Bangalore)", phone: "+91 80 2580 0826", area: "Frazer Town", type: "Early Intervention" },
  { name: "Bubbles Centre (Bangalore)", phone: "+91 80 4091 8971", area: "Yelahanka", type: "Therapy & School" },
  { name: "AIISH (Mysuru)", phone: "+91 821 2502100", area: "Manasagangothri", type: "Clinical Services" },
  { name: "DIMHANS (Dharwad)", phone: "+91 836 2444343", area: "Hubli-Dharwad", type: "Mental Health Institute" },
  { name: "St. Agnes Special School (Mangaluru)", phone: "+91 824 2211233", area: "Bendorewell", type: "Developmental School" },
  { name: "CADABAMS (Bangalore)", phone: "+91 96111 94949", area: "JP Nagar", type: "Rehabilitation" },
  { name: "Spastic Society (Bangalore)", phone: "+91 80 4030 1888", area: "Indiranagar", type: "Physical & Mental Rehab" }
];

export default function Screening() {
  const [lang, setLang] = useState("en");
  const [domainIndex, setDomainIndex] = useState(-1); 
  const [inDetails, setInDetails] = useState(false); 
  const [detailPage, setDetailPage] = useState(0); 
  const [activeTip, setActiveTip] = useState(null); 
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [animatedScore, setAnimatedScore] = useState(0); 
  const [error, setError] = useState(null);
  const resultRef = useRef(null);

  const t = TRANSLATIONS[lang];

  const initialForm = {
    age: "", gender: "", jaundice: "0", autism_in_family: "0",
    ...Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`Q${i + 1}`, "0"]))
  };
  const [form, setForm] = useState(initialForm);

  const QUESTIONS_PER_PAGE = 2;

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const nextStep = () => {
    if (domainIndex === -1) {
      setDomainIndex(0); setInDetails(false); setActiveTip(null); return;
    }
    const domain = DOMAINS[domainIndex];
    if (!inDetails) {
      if (domain.gates.some(g => form[g.id] !== "0")) {
        setInDetails(true); setDetailPage(0);
      } else { moveToNextDomain(); }
    } else {
      const totalPages = Math.ceil(domain.details.length / QUESTIONS_PER_PAGE);
      if (detailPage + 1 < totalPages) { setDetailPage(detailPage + 1); }
      else { moveToNextDomain(); }
    }
  };

  const moveToNextDomain = () => {
    if (domainIndex + 1 < DOMAINS.length) {
      setDomainIndex(domainIndex + 1); setInDetails(false); setDetailPage(0); setActiveTip(null);
    } else { setDomainIndex(DOMAINS.length); setActiveTip(null); }
  };

  const prevStep = () => {
    if (domainIndex === DOMAINS.length) {
      setDomainIndex(DOMAINS.length - 1); setInDetails(true);
      const prevDomain = DOMAINS[DOMAINS.length - 1];
      setDetailPage(Math.ceil(prevDomain.details.length / QUESTIONS_PER_PAGE) - 1);
      return;
    }
    if (inDetails) {
      if (detailPage > 0) { setDetailPage(detailPage - 1); setActiveTip(null); }
      else { setInDetails(false); setActiveTip(null); }
    } else {
      if (domainIndex > 0) { setDomainIndex(domainIndex - 1); setInDetails(false); setActiveTip(null); }
      else { setDomainIndex(-1); setActiveTip(null); }
    }
  };

  const handleSubmit = async () => {
    setLoading(true); setError(null);
    try {
      const response = await fetch("http://127.0.0.1:5000/evaluate_isaa", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  React.useEffect(() => {
    if (result && result.total_score !== undefined) {
      let start = 0;
      const end = result.total_score;
      const duration = 1500; 
      const increment = end / (duration / 16);
      
      const timer = setInterval(() => {
        start += increment;
        if (start >= end) {
          setAnimatedScore(end);
          clearInterval(timer);
        } else {
          setAnimatedScore(Math.floor(start));
        }
      }, 16);
      return () => clearInterval(timer);
    }
  }, [result]);

  const downloadPDF = async () => {
    const canvas = await html2canvas(resultRef.current);
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const ratio = canvas.width / canvas.height;
    const imgHeight = (pageWidth - 20) / ratio;
    pdf.text("Autism Screening Analysis - Clinical Report", 10, 10);
    pdf.addImage(imgData, "PNG", 10, 20, pageWidth - 20, imgHeight);
    pdf.save(`Autism_Report_${form.age}Y.pdf`);
  };

  const getDomainData = () => {
    return DOMAINS.map(d => {
      const allQs = [...d.gates, ...d.details];
      const avg = allQs.reduce((acc, q) => acc + parseFloat(form[q.id]), 0) / allQs.length;
      // Clamp between 5 and 95 to ensure the graph is always visible and never touches the absolute edge
      const clampedScore = Math.max(5, Math.min(95, avg * 100));
      return { domain: d.id.toUpperCase(), score: clampedScore };
    });
  };

  const renderProgressBar = () => {
    if (domainIndex === -1 || result) return null;
    // Calculate total questions passed vs total (approx 40)
    // For simplicity, let's use Domain progress: (domainIndex / DOMAINS.length) * 100
    const progress = ((domainIndex) / DOMAINS.length) * 100;
    return (
      <div className="progress-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
        <span className="progress-text">Step {domainIndex + 1} of {DOMAINS.length}</span>
      </div>
    );
  };

  const renderContent = () => {
    if (domainIndex === -1) {
      return (
        <div className="fade-in">
          <div className="lang-toggle-bar">
            {["en", "kn", "hi"].map(l => (
              <button key={l} className={`lang-btn ${lang === l ? "active" : ""}`} onClick={() => setLang(l)}>{l.toUpperCase()}</button>
            ))}
          </div>
          <h2 className="section-label">{t.generalInfo}</h2>
          <div className="form-group"><label>{t.age}</label><input type="number" name="age" value={form.age} onChange={handleChange} required /></div>
          <div className="form-group">
            <label>{t.gender}</label>
            <select name="gender" value={form.gender} onChange={handleChange} required>
              <option value="">{t.select}</option><option value="Male">Male</option><option value="Female">Female</option>
            </select>
          </div>
          <div className="form-group"><label>{t.jaundice}</label><select name="jaundice" value={form.jaundice} onChange={handleChange}><option value="0">No</option><option value="1">Yes</option></select></div>
          <div className="form-group"><label>{t.familyHistory}</label><select name="autism_in_family" value={form.autism_in_family} onChange={handleChange}><option value="0">No</option><option value="1">Yes</option></select></div>
        </div>
      );
    }

    if (domainIndex >= 0 && domainIndex < DOMAINS.length) {
      const domain = DOMAINS[domainIndex];
      const questions = inDetails ? domain.details.slice(detailPage * QUESTIONS_PER_PAGE, (detailPage + 1) * QUESTIONS_PER_PAGE) : domain.gates;
      return (
        <div className="fade-in">
          <h2 className="section-label">{domain.title}</h2>
          {questions.map((q) => {
            const val = form[q.id];
            // If inverted: Always (1) for frontend display should reflect that 'Always' is the healthy choice (0 risk)
            // But we actually want to show 'Always' in the dropdown and have it map to 0 in the form state
            const getDisplayValue = (v) => {
              if (!q.isInverted) return v;
              // If val is 0 (Rarely in backend), it means 'Always' was selected in inverted UI
              // Wait, let's keep it simple:
              // For Inverted: Rarely (UI) -> 1 (State), Always (UI) -> 0 (State)
              if (v === "0") return "1"; // Always in state -> 0 in UI? No.
              if (v === "1") return "0"; // Rarely in state -> 1 in UI? No.
              return v;
            };

            const handleSelectChange = (e) => {
              let nextVal = e.target.value; // This is the UI value (0=Rarely, 1=Always)
              if (q.isInverted) {
                // Flip it for the state
                if (nextVal === "0") nextVal = "1";
                else if (nextVal === "1") nextVal = "0";
              }
              setForm({ ...form, [q.id]: nextVal });
            };

            // Calculate display value for the select:
            // If isInverted is true:
            // State 1 (High Risk) -> Should show Rarely (0) in UI
            // State 0 (Low Risk) -> Should show Always (1) in UI
            let displayVal = val;
            if (q.isInverted) {
              displayVal = val === "1" ? "0" : (val === "0" ? "1" : val);
            }

            return (
              <div key={q.id} className="form-group">
                <div className="q-header-row">
                  <label className="q-label-with-tip">
                    {q.text}
                  </label>
                  <button 
                    type="button" 
                    className={`info-icon-btn ${activeTip === q.id ? "active" : ""}`} 
                    onClick={() => setActiveTip(activeTip === q.id ? null : q.id)}
                    title="Click for more info"
                  >
                    ⓘ
                  </button>
                </div>
                
                {activeTip === q.id && (
                  <div className="question-tip-box fade-in">
                    <p className="tip-text">{q.tip}</p>
                  </div>
                )}

                <select name={q.id} value={displayVal} onChange={handleSelectChange} required>
                  <option value="0">{t.rarely}</option><option value="0.5">{t.sometimes}</option><option value="1">{t.always}</option>
                </select>
              </div>
            );
          })}
        </div>
      );
    }

    return (
      <div className="final-step fade-in">
        <h2 className="section-label">Review & Submit</h2>
        <p>You have completed the adaptive assessment. Click below to view the results.</p>
      </div>
    );
  };

  const getRecommendations = (risk) => {
    switch(risk) {
      case "High":
        return [
          "🎯 Schedule a clinical evaluation with a Developmental Pediatrician immediately.",
          "📑 Bring this Radar Analysis & ISAA Score to your specialist appointment.",
          "🧠 Look into Early Intervention (EI) programs like Speech or Occupational therapy.",
          "🏘️ Connect with specialized centers (see Karnataka Resources below)."
        ];
      case "Moderate":
        return [
          "👨‍⚕️ Consult your pediatrician about these screening results.",
          "⏳ Re-screen in 3-6 months to monitor developmental progress.",
          "🧩 Increase structured social-emotional play and interaction at home.",
          "📍 Visit a local early intervention center for a secondary check."
        ];
      default:
        return [
          "🌟 Continue monitoring standard developmental milestones.",
          "📅 Maintain routine pediatric check-ups.",
          "📚 Engage in diverse social and sensory play activities.",
          "🌱 No immediate clinical action is suggested based on this screening."
        ];
    }
  };

  if (result) {
    const scoreData = getDomainData();
    const recommendations = getRecommendations(result.risk_level);
    const statusClass = result.risk_level.toLowerCase();
    return (
      <div className="screening-container">
        <div className="result-card" ref={resultRef}>
          <h2 className="result-title">{t.resultTitle}</h2>
          <div className={`status-badge ${statusClass}`}>{result.risk_level === "High" ? t.highRisk : result.risk_level === "Moderate" ? t.moderateRisk : t.lowRisk}</div>
          
          <div className="analysis-grid">
            <div className="chart-container">
              <p className="chart-title">Domain Breakdown (RADAR)</p>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={scoreData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="domain" tick={{ fontSize: 10, fill: "#64748b" }} />
                  <Radar name="Score" dataKey="score" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.4} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="score-summary">
              <p className="score-total-wrapper">
                <b>{t.scoreLabel}:</b> <span className="animated-score">{animatedScore}</span>
              </p>
              <div className="result-info"><p>This assessment is based on the ISAA clinical standard. A score of {result.total_score} indicates <b>{result.risk_level} Risk</b> levels.</p></div>
            </div>
          </div>

          <div className="recommendation-section">
            <h3>{t.recommendation}</h3>
            <ul className="suggestions-list">
              {recommendations.map((rec, i) => (
                <li key={i}>{rec}</li>
              ))}
            </ul>
          </div>

          <div className="resources-section">
            <h3>{t.resources}</h3>
            <div className="centers-grid">
              {KARNATAKA_CENTERS.map((c, i) => (
                <div key={i} className="center-card">
                  <strong>{c.name}</strong>
                  <p>📍 {c.area} | 📞 {c.phone}</p>
                  <span className="type-tag">{c.type}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="actions-bar">
            <button className="reset-btn" onClick={() => window.location.reload()}>{t.startOver}</button>
            <button className="pdf-btn" onClick={downloadPDF}>{t.downloadPDF}</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="screening-container">
      <div className="screening-card">
        <h1 className="screening-title">{t.title}</h1>
        {renderProgressBar()}
        {loading ? <div className="loading-container"><div className="spinner"></div><p>Generating Analysis...</p></div> : (
          <form onSubmit={(e) => e.preventDefault()} className="screening-form">
            {renderContent()}
            <div className="nav-buttons">
              {domainIndex >= 0 && <button type="button" className="prev-btn" onClick={prevStep}>{t.back}</button>}
              {domainIndex < DOMAINS.length ? (
                <button type="button" className="next-btn" onClick={nextStep}>{t.next}</button>
              ) : (
                <button type="button" className="submit-btn" onClick={handleSubmit}>{t.submit}</button>
              )}
            </div>
          </form>
        )}
        {error && <div className="error-msg">⚠️ {error}</div>}
      </div>
    </div>
  );
}
