# CS181DV Final Project: Interactive AI Content Visualization

**Author**: Aiko Kato  
**Date**: May 7, 2025

## Overview

This project analyzes the global impact of AI-generated content across countries, industries, and AI tools from 2020 to 2025. 
It includes a complete data processing pipeline and an interactive web-based dashboard built with Dash and Plotly.

The dashboard provides visual insights into trends such as AI adoption, content volume, tool usage, regulation, consumer trust, and industry-specific impacts. It supports filtering, linked views, and advanced visualizations like network graphs.

---

## Project Structure

project/
│
├── docs/
│ ├── proposal.pdf
│ └── lightning_talk.pdf
│
├── src/
│ ├── data_processing/
│ │ ├── __init__.py  # Allow the directory to be treated as a Python package
│ │ ├── process_data.py  # Data cleaning, aggregation, and caching
│ │ └── Global_AI_Content_Impact_Dataset.csv
│ │
│ └── visualization/
│   └── core.py  # Dash app and all visualization logic
│
├── tests/
│ └── test_visualization.py  # Unit tests for data processing
│
├── requirements.txt  # Required Python packages
└── README.md # Project overview and instructions

---

## Features

### Data Processing Pipeline
- Loads real-world CSV dataset
- Cleans and validates fields (text and numeric)
- Aggregates statistics by country, industry, and AI tool
- Saves cleaned and summary datasets to disk for reuse

### Core Visualizations
- Choropleth map of AI metrics by country and year
- Time-series line charts of industry trends
- Bar charts for AI-generated content and trust by regulation
- Scatter plots comparing AI adoption with job loss/revenue increase
- Heatmap of content volume across industries and years

### Advanced Features
- Interactive network graph showing relationships between industries and AI tools
- Toggleable legend groups and custom hover text
- Linked views (scatter + line charts) for drill-down exploration

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher installed
- Modern web browser (Chrome, Firefox, Edge)
- Dash and Plotly libraries (install via `requirements.txt`)
- Install all packages: `pip install -r requirements.txt`

---

## Installation Instructions

#### 1. Preprocess the dataset:
```bash
python src/data_processing/process_data.py
```

#### 2. Launch the Dash app:
```bash
python src/visualization/core.py
```

#### 3. Open your web browser and go to:
```bash
http://127.0.0.1:8050/
```
You should now see the interactive AI Content Visualization dashboard.

---

## Testing

### Run Unit Tests
```bash
python -m unittest tests.test_visualization
```

---

## Task Completion Overview  
This project implements all required features for CS181DV Final Project: Interactive Data Visualization System.

### Task 1 – Data Processing Pipeline
✔ **Load and parse raw dataset** → Read from CSV using pandas  
✔ **Clean and validate fields** → Removed nulls, standardized text, coerced numerics  
✔ **Aggregate summary stats** → Grouped by country, industry, and AI tool  
✔ **Cache results** → Saved cleaned and summary CSVs to disk  

### Task 2 – Core Visualizations
✔ **Choropleth maps** → Visualize AI metrics by country and year  
✔ **Line charts** → Track industry adoption trends over time  
✔ **Bar charts** → Content volume, tool usage, and regulation trust levels  
✔ **Scatter plots** → Compare adoption vs. job loss/revenue  
✔ **Heatmap** → Content volume by industry and year  

### Task 3 – Advanced Features
✔ **Network graph** → Visualize relationships between AI tools and industries  
✔ **Toggleable node groups** → Interactive legend enables industry/tool filtering  
✔ **Linked views** → Scatter + line charts filter trends by selection
Note: I did not implement 3D visualization, as I already incorporated advanced techniques and the professor confirmed that 3D is optional. Real-time updates were not included because the dataset does not offer live data.

### Task 4 – Integration & Testing
✔ **Smooth integration** → Data pipeline feeds directly into Dash dashboard  
✔ **Performance optimization** → Preprocessing avoids recomputation  
✔ **Unit tests included** → Tests validate cleaning and summary generation  
✔ **Well-documented code** → Docstrings and comments throughout  

---

## License
This project was completed by Aiko Kato as part of CS181DV at Pomona College.

---
