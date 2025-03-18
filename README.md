# 📊 Interactive Data Explorer

A powerful Streamlit-based web application for exploratory data analysis that allows users to upload their own datasets or use sample data for interactive visualization and analysis.

## Features

- **File Upload**: Support for CSV, Excel, JSON, and other common data formats
- **Data Overview**: Quick summary statistics and data preview
- **Interactive Visualizations**: Generate various charts and plots
- **Data Analysis**: Correlation analysis, distribution analysis, and more
- **Sample Data**: Option to explore with pre-generated sample data

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-explorer.git
cd data-explorer

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```shell
streamlit run app.py
```

This will start the application on `localhost:8501` (by default).

## Project Structure

```
data-explorer/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── .streamlit/            # Streamlit configuration
│   └── config.toml        # Config settings
├── backend/               # Backend modules
│   └── data.py            # Data loading
└── frontend/              # Frontend components
    ├── sidebar.py         # Sidebar with upload and controls
    ├── show_data.py       # Data overview components
    └── data_analysis.py   # Analysis and visualization tabs
```

## Requirements

- 3.7 <= Python <= 3.10 (I used 3.10)

See `requirements.txt` for the complete list of dependencies.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
