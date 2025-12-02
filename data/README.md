# Data Directory

## Data Sources

**Primary Source:** EDP Open Data - Wind Turbine SCADA and Event Data  
**URL:** https://www.edp.com/en/innovation/data  
**License:** Open Data License

### Datasets Used

1. **SCADA Data** (Sensor readings at 10-minute intervals)
   - Farm A: (81 sensors, 6 months)
   - Farm B: (257 sensors, 1 year)
   - Farm C: (952 sensors, 8 months)

2. **Event Data** (Failure timestamps and descriptions)
   - Farm A: `events_farm_a.csv` (12 failures)
   - Farm B: `events_farm_b.csv` (6 failures)
   - Farm C: `events_farm_c.csv` (27 failures)

3. **Sensor Mapping** (Unified categorization)
   - `sensor_mapping_v2.csv` (283 sensors → 16 unified categories)

## Data Structure

### SCADA Format
```
time_stamp, asset_id, status_type_id, power_30_avg, sensor_5_min, ...
2020-01-01 00:00:00, 0, 0, 125000.5, 5.2, ...
```

### Events Format
```
event_id, asset_id, event_start, event_end, event_label, event_description
0, 0, 2020-03-15 14:23:00, 2020-03-16 09:45:00, anomaly, Generator bearing
```

### Sensor Mapping Format
```
sensor_name, description, unit, farm, primary group, secondary group
sensor_18, Generator RPM variability, rpm, A, gearbox, mechanical
```
### **4. requirements.txt**
```
# Core
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
tqdm>=4.62.0
python-dateutil>=2.8.2

# Optional: Streamlit (for web app)
streamlit>=1.10.0

# Optional: Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# Optional: Documentation
mkdocs>=1.3.0
mkdocs-material>=8.0.0
```

---

### **5. LICENSE**
```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.