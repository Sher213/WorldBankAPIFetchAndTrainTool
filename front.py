from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from google import genai
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from get_data import translate_indicators, get_data_df
from generate_model import model_pipeline

load_dotenv()
client = genai.Client()

# Mapping of countries to ISO 3166-1 alpha-2 codes for World Bank API
COUNTRIES = {
    "Afghanistan": "AF",
    "Albania": "AL",
    "Algeria": "DZ",
    "Andorra": "AD",
    "Angola": "AO",
    "Antigua and Barbuda": "AG",
    "Argentina": "AR",
    "Armenia": "AM",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Bahamas": "BS",
    "Bahrain": "BH",
    "Bangladesh": "BD",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Belize": "BZ",
    "Benin": "BJ",
    "Bhutan": "BT",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Brazil": "BR",
    "Brunei": "BN",
    "Bulgaria": "BG",
    "Burkina Faso": "BF",
    "Burundi": "BI",
    "Cabo Verde": "CV",
    "Cambodia": "KH",
    "Cameroon": "CM",
    "Canada": "CA",
    "Central African Republic": "CF",
    "Chad": "TD",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "Comoros": "KM",
    "Congo, Republic of the": "CG",
    "Congo, Democratic Republic of the": "CD",
    "Costa Rica": "CR",
    "Côte d'Ivoire": "CI",
    "Croatia": "HR",
    "Cuba": "CU",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Djibouti": "DJ",
    "Dominica": "DM",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Equatorial Guinea": "GQ",
    "Eritrea": "ER",
    "Estonia": "EE",
    "Eswatini": "SZ",
    "Ethiopia": "ET",
    "Fiji": "FJ",
    "Finland": "FI",
    "France": "FR",
    "Gabon": "GA",
    "Gambia": "GM",
    "Georgia": "GE",
    "Germany": "DE",
    "Ghana": "GH",
    "Greece": "GR",
    "Grenada": "GD",
    "Guatemala": "GT",
    "Guinea": "GN",
    "Guinea-Bissau": "GW",
    "Guyana": "GY",
    "Haiti": "HT",
    "Honduras": "HN",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Jordan": "JO",
    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kiribati": "KI",
    "Kuwait": "KW",
    "Kyrgyzstan": "KG",
    "Laos": "LA",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Lesotho": "LS",
    "Liberia": "LR",
    "Libya": "LY",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Madagascar": "MG",
    "Malawi": "MW",
    "Malaysia": "MY",
    "Maldives": "MV",
    "Mali": "ML",
    "Malta": "MT",
    "Marshall Islands": "MH",
    "Mauritania": "MR",
    "Mauritius": "MU",
    "Mexico": "MX",
    "Micronesia": "FM",
    "Moldova": "MD",
    "Monaco": "MC",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Morocco": "MA",
    "Mozambique": "MZ",
    "Myanmar": "MM",
    "Namibia": "NA",
    "Nauru": "NR",
    "Nepal": "NP",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Niger": "NE",
    "Nigeria": "NG",
    "North Korea": "KP",
    "North Macedonia": "MK",
    "Norway": "NO",
    "Oman": "OM",
    "Pakistan": "PK",
    "Palau": "PW",
    "Panama": "PA",
    "Papua New Guinea": "PG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Romania": "RO",
    "Russia": "RU",
    "Rwanda": "RW",
    "Saint Kitts and Nevis": "KN",
    "Saint Lucia": "LC",
    "Saint Vincent and the Grenadines": "VC",
    "Samoa": "WS",
    "San Marino": "SM",
    "São Tomé and Príncipe": "ST",
    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "Seychelles": "SC",
    "Sierra Leone": "SL",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Solomon Islands": "SB",
    "Somalia": "SO",
    "South Africa": "ZA",
    "South Korea": "KR",
    "South Sudan": "SS",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sudan": "SD",
    "Suriname": "SR",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Syria": "SY",
    "Tajikistan": "TJ",
    "Tanzania": "TZ",
    "Thailand": "TH",
    "Timor-Leste": "TL",
    "Togo": "TG",
    "Tonga": "TO",
    "Trinidad and Tobago": "TT",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Turkmenistan": "TM",
    "Tuvalu": "TV",
    "Uganda": "UG",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
    "Vanuatu": "VU",
    "Vatican City": "VA",
    "Venezuela": "VE",
    "Vietnam": "VN",
    "Yemen": "YE",
    "Zambia": "ZM",
    "Zimbabwe": "ZW"
}

TOP_100_WORLD_BANK_INDICATORS = [
    "GDP (current US$)",
    "GDP (constant 2015 US$)",
    "GDP growth (annual %)",
    "GDP per capita (current US$)",
    "GDP per capita growth (annual %)",
    "GNI (current US$)",
    "GNI per capita (current US$)",
    "Gross capital formation (% of GDP)",
    "Gross savings (% of GDP)",
    "Government final consumption expenditure (% of GDP)",
    "Inflation, consumer prices (annual %)",
    "Consumer price index (2010=100)",
    "GDP deflator (annual %)",
    "Exports of goods and services (% of GDP)",
    "Imports of goods and services (% of GDP)",
    "Trade (% of GDP)",
    "Current account balance (% of GDP)",
    "Foreign direct investment, net inflows (current US$)",
    "External debt stocks (current US$)",
    "Population, total",
    "Population growth (annual %)",
    "Urban population (% of total)",
    "Rural population (% of total)",
    "Population ages 0-14 (% of total)",
    "Population ages 65 and above (% of total)",
    "Life expectancy at birth, total (years)",
    "Fertility rate, total (births per woman)",
    "Unemployment, total (% of labor force)",
    "Labor force participation rate, total (%)",
    "Employment to population ratio, total (%)",
    "Vulnerable employment (% of total employment)",
    "Poverty headcount ratio at $2.15 a day (%)",
    "Gini index",
    "Income share held by lowest 20%",
    "Income share held by highest 20%",
    "Health expenditure (% of GDP)",
    "Current health expenditure (current US$)",
    "Mortality rate, under-5 (per 1,000 live births)",
    "Maternal mortality ratio (per 100,000 live births)",
    "Physicians (per 1,000 people)",
    "Hospital beds (per 1,000 people)",
    "Immunization, measles (% of children)",
    "Literacy rate, adult total (%)",
    "School enrollment, primary (% gross)",
    "School enrollment, secondary (% gross)",
    "School enrollment, tertiary (% gross)",
    "Education expenditure (% of GDP)",
    "Pupil-teacher ratio, primary",
    "Access to electricity (% of population)",
    "Electric power consumption (kWh per capita)",
    "Energy use (kg of oil equivalent per capita)",
    "Renewable energy consumption (% of total)",
    "CO2 emissions (metric tons per capita)",
    "CO2 emissions (kt)",
    "Forest area (% of land area)",
    "Methane emissions (kt of CO2 equivalent)",
    "Nitrous oxide emissions (kt of CO2 equivalent)",
    "Individuals using the Internet (% of population)",
    "Mobile cellular subscriptions (per 100 people)",
    "Fixed broadband subscriptions (per 100 people)",
    "Domestic credit to private sector (% of GDP)",
    "Broad money (% of GDP)",
    "Lending interest rate (%)",
    "Real interest rate (%)",
    "Government effectiveness (estimate)",
    "Rule of law (estimate)",
    "Control of corruption (estimate)",
    "Political stability (estimate)",
    "Regulatory quality (estimate)",
    "Voice and accountability (estimate)",
    "Agriculture, forestry, and fishing (% of GDP)",
    "Arable land (% of land area)",
    "Cereal yield (kg per hectare)",
    "Industry (including construction) (% of GDP)",
    "Manufacturing (% of GDP)",
    "Services (% of GDP)",
    "Total reserves (current US$)",
    "Total debt service (% of exports)",
    "Adolescent fertility rate (births per 1,000 women)",
    "Age dependency ratio (% of working-age population)",
    "Population in urban agglomerations >1 million (%)",
    "PM2.5 air pollution (micrograms per cubic meter)"
]

def display_model_stats(stats, problem_type="regression"):
    st.subheader("📊 Model Performance Statistics")

    if problem_type == "regression":
        st.metric("Train R²", round(stats['train_r2'], 3))
        st.metric("Test R²", round(stats['test_r2'], 3))
        st.metric("Train MSE", round(stats['train_mse'], 3))
        st.metric("Test MSE", round(stats['test_mse'], 3))

        # Plot predictions vs actual for test set
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(range(len(stats['y_test'])), stats['y_test'], label="Actual", alpha=0.7)
        ax.scatter(range(len(stats['y_test'])), stats['y_pred_test'], label="Predicted", alpha=0.7)
        ax.set_title("Actual vs Predicted (Test Set)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Target")
        ax.legend()
        st.pyplot(fig)

    else:  # classification
        st.metric("Train Accuracy", round(stats['train_accuracy'], 3))
        st.metric("Test Accuracy", round(stats['test_accuracy'], 3))

        # Show classification report table
        st.write("Classification Report (Test Set)")
        import pandas as pd
        report_df = pd.DataFrame(stats['classification_report']).transpose()
        st.dataframe(report_df)

        # Plot training history if available
        if 'history' in stats:
            fig, ax = plt.subplots(1, 2, figsize=(12,4))

            # Loss
            ax[0].plot(stats['history']['loss'], label="Train Loss")
            ax[0].plot(stats['history'].get('val_loss', []), label="Val Loss")
            ax[0].set_title("Model Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].legend()

            # Accuracy
            if 'accuracy' in stats['history']:
                ax[1].plot(stats['history']['accuracy'], label="Train Accuracy")
                ax[1].plot(stats['history'].get('val_accuracy', []), label="Val Accuracy")
                ax[1].set_title("Model Accuracy")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Accuracy")
                ax[1].legend()

            st.pyplot(fig)

# ---------------------------
# Initialize session state
# ---------------------------
if "selected_indicator" not in st.session_state:
    st.session_state.selected_indicator = []

if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = []

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

if "model" not in st.session_state:
    st.session_state.stats = {}

if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []

if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}

if "scaler" not in st.session_state:
    st.session_state.scaler = None

st.set_page_config(page_title="World Bank Dashboard", layout="centered")
st.title("🌍 World Bank Indicators Dashboard")

# ---------------------------
# Step 1: Country & Indicator Selection
# ---------------------------
st.subheader("Select Countries")
country = st.selectbox("Choose a country:", sorted(COUNTRIES.keys()))

if st.button("Add Countries"):
    if country not in st.session_state.selected_countries:
        st.session_state.selected_countries.append(country)

st.success("No countries selected yet!" if not st.session_state.selected_countries else
           f"Selected countries: {', '.join(st.session_state.selected_countries)}")

st.subheader("Select Indicators")
indicator = st.selectbox("Choose an indicator:", sorted(TOP_100_WORLD_BANK_INDICATORS))

if st.button("Add Indicator"):
    if indicator not in st.session_state.selected_indicator:
        st.session_state.selected_indicator.append(indicator)
    else:
        st.warning(f"'{indicator}' is already in your list!")

st.write("Indicators selected:")
if not st.session_state.selected_indicator:
    st.info("No indicators yet!")
else:
    st.success("\n".join(f"{i+1}. {ind}" for i, ind in enumerate(st.session_state.selected_indicator)))

if st.button("Fetch Data for Selected Indicators"):
    if not st.session_state.selected_indicator:
        st.error("Please select at least one indicator.")
    else:
        with st.spinner("Fetching data..."):
            translated = translate_indicators(st.session_state.selected_indicator)
            st.session_state.data = get_data_df([COUNTRIES[c] for c in st.session_state.selected_countries],
                                                translated, "2000:2020")
        st.success(f"Fetched data for {len(st.session_state.selected_indicator)} indicators!")
        st.dataframe(st.session_state.data)

# ---------------------------
# Step 2: Hypothesis Chatbot
# ---------------------------
if not st.session_state.data.empty:
    st.subheader("🤖 Hypothesis Question Chatbot for Datasets")
    st.write("Columns in dataset:")
    st.write(st.session_state.data.columns.tolist())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_prompt = st.text_input("Enter your question or prompt for hypothesis generation:")

    if st.button("Generate Hypothesis Questions") and user_prompt:
        with st.spinner("Generating hypothesis questions..."):
            dataset_snippet = st.session_state.data.head(5).to_csv(index=False)
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{user_prompt}\n\nHere is a snippet of your dataset:\n{dataset_snippet}"
            )
            ai_text = response.text if response else "No response from model."
            st.session_state.chat_history.append({"user": user_prompt, "ai": ai_text})

    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")
        st.markdown("---")

# ---------------------------
# Step 3: Train Model
# ---------------------------
if not st.session_state.data.empty:
    st.subheader("Train a Machine Learning Model")

    problem_type = st.selectbox("Select problem type:", ["Regression", "Classification"])
    target_col = st.selectbox("Select target column:",
                              st.session_state.data.drop(columns=['country', 'date'], errors='ignore').columns) \
        if problem_type == "Regression" else "country"

    if st.button("Train Model"):
        st.info(f"Training {problem_type} model with target '{target_col}'...")
        st.session_state.feature_cols = st.session_state.data.columns.tolist()
        st.session_state.stats = model_pipeline(st.session_state.data, target_col, problem_type.lower(),
                       num_classes=st.session_state.data['country'].nunique())
        st.success("Model training complete!")
    
    # Display metrics & plots
    display_model_stats(st.session_state.stats, problem_type.lower())
# ---------------------------
# Step 5: Dataset Visualizations
# ---------------------------
if not st.session_state.data.empty:
    st.subheader("Dataset Visualizations")

    # Histogram for numeric columns
    numeric_cols = st.session_state.data.select_dtypes(include=np.number).columns.tolist()
    st.write("Numeric Columns Histogram")
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8,3))
        sns.histplot(st.session_state.data[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Scatter plot for any two numeric columns
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis column for scatter plot:", numeric_cols)
        y_col = st.selectbox("Y-axis column for scatter plot:", numeric_cols, index=1)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=st.session_state.data[x_col], y=st.session_state.data[y_col], ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)