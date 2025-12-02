# Wind Turbine Failure Prediction
### Systematic Methodology with 67% Recall, 100% Precision

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-FF4B4B)](https://windfarm.streamlit.app/)

**Portfolio Project** | Machine Learning for Predictive Maintenance 
---

## Project Overview

Developed a failure prediction system for wind turbines that detects 67% of failures 24 hours in advance with 100% precision on validation farms. Validated systematic methodology across three independent wind farms with different turbine models (81 to 952 sensors).

**Key Innovation:** Unified multi-farm model using semantic sensor mapping enables knowledge transfer between heterogeneous turbine configurations, doubling performance on limited-data scenarios (33% → 67% recall).

### Quick Stats

| Metric | Farm A (Dev) | Farm B (Limited) | Farm C (Validation) |
|--------|--------------|------------------|---------------------|
| **Environment** | Onshore Portugal | Offshore Germany | Offshore Germany |
| **Sensors** | 81 | 257 | 952 |
| **Failures** | 12 | 6 | 27 |
| **Recall** | 67% → 58%| 33% → 50% | 67% → 56% |
| **Precision** | 89% → 100% | 100% → 75% | 100% |

*\* Unified model result (farm-specific: 33%)*

---
## Key Results

### Validation Across Three Independent Farms

✅ **Consistent Recall:** 67% on adequately-sized datasets (Farms A & C)  
✅ **Perfect Precision:** 100% on validation farms (B & C) - zero false alarms  
✅ **High Accuracy:** 98% overall classification accuracy  
✅ **Advance Warning:** 24-hour lead time for maintenance scheduling  

### Unified Model Performance

**Problem:** Farm-specific models fail with limited data (n=6 failures → 33% recall)

**Solution:** Semantic sensor mapping enables cross-farm training

**Result:** Combined training (n=45 failures) → 67% recall on Farm B

| Approach | Farm B Recall | Farm A/C Recall | Precision |
|----------|---------------|-----------------|-----------|
| Farm-Specific | 33% ❌ | 67% | 89-100% |
| **Unified Model** | **67%** ✅ | 59-67% | 97% |
| **Improvement** | **+34%** | -0 to -8% | Maintained |

---

## Methodology Highlights

### 1. Data Validation
- Verify SCADA structure and completeness
- Identify power sensor and production periods
- Critical discovery: `status==0 AND power>0.1` for true normal operation

### 2. Temporal Pattern Analysis
- Calculate time between when the failure was logged and the last know normal state
- Identify two patterns: 7-day gap vs. immediate logging
- Use last normal production time as prediction target

### 3. Feature Engineering
- **Window aggregation:** Mean, std, trend over 24h
- **Scales gracefully:** 81 → 952 sensors handled uniformly
- **Prevents leakage:** No future information in features

### 4. Feature Selection
- **Phase A:** Cohen's d for discriminative power (d > 0.7)
- **Phase B:** Iterative correlation filtering (r < 0.9)
- **Result:** 15-17 diverse features per farm

### 5. Model Evaluation
- Random Forest (100 trees, max_depth=5, balanced weights)
- Leave-One-Out cross-validation
- Per-event probability tracking

---

## Key Learnings

### What Worked

**Validation-first approach:** Data quality issues caught early  
**Window aggregation:** More scalable than rolling features  
**Precision-focused:** Zero false alarms on validation maintains trust  
**Systematic methodology:** Transfers successfully across farms  

### Challenges Discovered

**Smooth degradation:** Event 40 (declining RPM) missed by aggregated features  
**Temporal patterns:** Different failure logging practices across farms  
**Sample size:** Minimum 10-12 failures needed for 60%+ recall if not unified  
**Class Imbalance:** Infrequent failures produced < 1% data during failed states  
**Feature redundancy:** Massive correlation (214 pairs r>0.9 in top 50 influential features)  

---
## Technical Stack

**Languages & Core:**
- Python 3.8+
- Pandas, NumPy, SciPy

**Machine Learning:**
- scikit-learn (Random Forest, cross-validation)
- Feature engineering & selection

**Visualization:**
- Matplotlib, Seaborn
- Plotly (interactive Streamlit charts)

**Deployment:**
- Streamlit (web application)
- Production pipeline design

---

## 🌐 Live Demo

**Interactive Streamlit App:** [https://windfarm.streamlit.app/](https://windfarm.streamlit.app/)

Explore:
- Methodology walkthrough
- Per-farm performance details
- Feature importance analysis
- Model performance metrics
- Deployment calculator
- Unified model comparison

---

## Documentation

### Complete Guides

- [**Full Methodology**](docs/methodology.md) - 15,000-word detailed explanation
- [**Unified Model**](docs/unified_model.md) - Cross-farm learning approach

### Quick Links

- [Data Sources](data/README.md)
- [Example Notebooks](notebooks/)
- [Model Results](results/)

---

## Contributing

This is a learning project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Keith**

- LinkedIn: [linkedin.com/in/kcockerham](https://linkedin.com/in/kcockerham)

---

## Acknowledgments

- **Data Source:** EDP Open Data - Wind Turbine SCADA Dataset
- **Inspiration:** Real-world industrial predictive maintenance challenges
- **Purpose:** Data science project

*Last Updated: December 2025*