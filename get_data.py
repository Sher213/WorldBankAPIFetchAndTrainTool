import numpy as np
import requests
import pandas as pd
import time

# This function translates user-friendly indicator names into World Bank API codes.
def translate_indicators(user_fields: list[str]=["GDP (current US$)", "Population, total"]) -> list[str]:
    indicators = []

    TOP_100_WORLD_BANK_INDICATORS = {
        # --- GDP & Growth ---
        "GDP (current US$)": "NY.GDP.MKTP.CD",
        "GDP (constant 2015 US$)": "NY.GDP.MKTP.KD",
        "GDP growth (annual %)": "NY.GDP.MKTP.KD.ZG",
        "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
        "GDP per capita growth (annual %)": "NY.GDP.PCAP.KD.ZG",
        "GNI (current US$)": "NY.GNP.MKTP.CD",
        "GNI per capita (current US$)": "NY.GNP.PCAP.CD",
        "Gross capital formation (% of GDP)": "NE.GDI.TOTL.ZS",
        "Gross savings (% of GDP)": "NY.GNS.ICTR.ZS",
        "Government final consumption expenditure (% of GDP)": "NE.CON.GOVT.ZS",

        # --- Inflation & Prices ---
        "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
        "Consumer price index (2010=100)": "FP.CPI.TOTL",
        "GDP deflator (annual %)": "NY.GDP.DEFL.KD.ZG",

        # --- Trade ---
        "Exports of goods and services (% of GDP)": "NE.EXP.GNFS.ZS",
        "Imports of goods and services (% of GDP)": "NE.IMP.GNFS.ZS",
        "Trade (% of GDP)": "NE.TRD.GNFS.ZS",
        "Current account balance (% of GDP)": "BN.CAB.XOKA.GD.ZS",
        "Foreign direct investment, net inflows (BoP, current US$)": "BX.KLT.DINV.CD.WD",
        "Foreign direct investment, net inflows (% of GDP)": "BX.KLT.DINV.WD.GD.ZS",
        "External debt stocks (current US$)": "DT.DOD.DECT.CD",

        # --- Population ---
        "Population, total": "SP.POP.TOTL",
        "Population growth (annual %)": "SP.POP.GROW",
        "Urban population (% of total)": "SP.URB.TOTL.IN.ZS",
        "Rural population (% of total)": "SP.RUR.TOTL.ZS",
        "Population ages 0-14 (% of total)": "SP.POP.0014.TO.ZS",
        "Population ages 65 and above (% of total)": "SP.POP.65UP.TO.ZS",
        "Life expectancy at birth, total (years)": "SP.DYN.LE00.IN",
        "Fertility rate, total (births per woman)": "SP.DYN.TFRT.IN",

        # --- Labor ---
        "Unemployment, total (% of total labor force)": "SL.UEM.TOTL.ZS",
        "Labor force participation rate, total (%)": "SL.TLF.CACT.ZS",
        "Employment to population ratio, total (%)": "SL.EMP.TOTL.SP.ZS",
        "Vulnerable employment (% of total employment)": "SL.EMP.VULN.ZS",

        # --- Poverty & Inequality ---
        "Poverty headcount ratio at $2.15 a day (%)": "SI.POV.DDAY",
        "Gini index": "SI.POV.GINI",
        "Income share held by lowest 20%": "SI.DST.FRST.20",
        "Income share held by highest 20%": "SI.DST.05TH.20",

        # --- Health ---
        "Health expenditure (% of GDP)": "SH.XPD.CHEX.GD.ZS",
        "Current health expenditure (current US$)": "SH.XPD.CHEX.CD",
        "Mortality rate, under-5 (per 1,000 live births)": "SH.DYN.MORT",
        "Maternal mortality ratio (per 100,000 live births)": "SH.STA.MMRT",
        "Physicians (per 1,000 people)": "SH.MED.PHYS.ZS",
        "Hospital beds (per 1,000 people)": "SH.MED.BEDS.ZS",
        "Immunization, measles (% of children ages 12-23 months)": "SH.IMM.MEAS",

        # --- Education ---
        "Literacy rate, adult total (% ages 15 and above)": "SE.ADT.LITR.ZS",
        "School enrollment, primary (% gross)": "SE.PRM.ENRR",
        "School enrollment, secondary (% gross)": "SE.SEC.ENRR",
        "School enrollment, tertiary (% gross)": "SE.TER.ENRR",
        "Education expenditure (% of GDP)": "SE.XPD.TOTL.GD.ZS",
        "Pupil-teacher ratio, primary": "SE.PRM.ENRL.TC.ZS",

        # --- Energy ---
        "Access to electricity (% of population)": "EG.ELC.ACCS.ZS",
        "Electric power consumption (kWh per capita)": "EG.USE.ELEC.KH.PC",
        "Energy use (kg of oil equivalent per capita)": "EG.USE.PCAP.KG.OE",
        "Renewable energy consumption (% of total final energy consumption)": "EG.FEC.RNEW.ZS",

        # --- Environment ---
        "CO2 emissions (metric tons per capita)": "EN.ATM.CO2E.PC",
        "CO2 emissions (kt)": "EN.ATM.CO2E.KT",
        "Forest area (% of land area)": "AG.LND.FRST.ZS",
        "Methane emissions (kt of CO2 equivalent)": "EN.ATM.METH.KT.CE",
        "Nitrous oxide emissions (kt of CO2 equivalent)": "EN.ATM.NOXE.KT.CE",

        # --- Infrastructure ---
        "Individuals using the Internet (% of population)": "IT.NET.USER.ZS",
        "Mobile cellular subscriptions (per 100 people)": "IT.CEL.SETS.P2",
        "Fixed broadband subscriptions (per 100 people)": "IT.NET.BBND.P2",

        # --- Financial Sector ---
        "Domestic credit to private sector (% of GDP)": "FS.AST.PRVT.GD.ZS",
        "Broad money (% of GDP)": "FM.LBL.BMNY.GD.ZS",
        "Lending interest rate (%)": "FR.INR.LEND",
        "Real interest rate (%)": "FR.INR.RINR",

        # --- Governance ---
        "Government effectiveness (estimate)": "GE.EST",
        "Rule of law (estimate)": "RL.EST",
        "Control of corruption (estimate)": "CC.EST",
        "Political stability (estimate)": "PV.EST",
        "Regulatory quality (estimate)": "RQ.EST",
        "Voice and accountability (estimate)": "VA.EST",

        # --- Agriculture ---
        "Agriculture, forestry, and fishing (% of GDP)": "NV.AGR.TOTL.ZS",
        "Arable land (% of land area)": "AG.LND.ARBL.ZS",
        "Cereal yield (kg per hectare)": "AG.YLD.CREL.KG",

        # --- Industry ---
        "Industry (including construction) (% of GDP)": "NV.IND.TOTL.ZS",
        "Manufacturing (% of GDP)": "NV.IND.MANF.ZS",

        # --- Services ---
        "Services (% of GDP)": "NV.SRV.TOTL.ZS",

        # --- Debt & Reserves ---
        "Total reserves (includes gold, current US$)": "FI.RES.TOTL.CD",
        "Total debt service (% of exports)": "DT.TDS.DECT.EX.ZS",

        # --- Demographics advanced ---
        "Adolescent fertility rate (births per 1,000 women ages 15-19)": "SP.ADO.TFRT",
        "Age dependency ratio (% of working-age population)": "SP.POP.DPND",

        # --- Climate & Risk ---
        "Population in urban agglomerations >1 million (%)": "EN.URB.MCTY.TL.ZS",
        "PM2.5 air pollution (micrograms per cubic meter)": "EN.ATM.PM25.MC.M3",
    }

    for field in user_fields:
        if field in TOP_100_WORLD_BANK_INDICATORS:
            indicators.append(TOP_100_WORLD_BANK_INDICATORS[field])

    return indicators

# This function retrieves data from the World Bank API based on specified indicators and date range.
def get_data_df(countries:list[str] = ["CA"], indicators: list[str] = ["NY.GDP.MKTP.CD"], date_range: str="2000:2020") -> list[dict]:
    data = []
    df = pd.DataFrame()
    
    base_url = "https://api.worldbank.org/v2/"
    
    for c in countries:
        for ind in indicators:
            url = f"{base_url}country/{c}/indicator/{ind}?date={date_range}&format=json"

            time.sleep(1)  # Sleep for 2 seconds to avoid hitting API rate limits

            response = requests.get(url)
            print(url)
            if response.status_code == 200:
                data.append(response.json())
            else:
                print(f"Error: {response.status_code}")
                return None
        
        rows = []
        for obs in data:  # Skip the first element which contains metadata
            if len(obs) < 2:
                continue

            obs = obs[1]
            for o in obs:
                rows.append({
                    "date": o["date"],
                    "country": o["country"]["value"],
                    "indicator": o["indicator"]["value"],
                    "value": o["value"]
            })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Format dataframe
    if not df.empty:
        for i in df['indicator'].unique():
            df[i] = df[df['indicator']==i]['value']
        df = df.drop(columns=['indicator', 'value']).drop_duplicates().reset_index(drop=True)
        
        df = df.groupby(['date', 'country'], as_index=False).agg(lambda x: next((v for v in x if pd.notna(v)), np.nan))

    return df