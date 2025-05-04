import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Crime Data EDA Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("train0.csv")
    df['Date_Occurred'] = pd.to_datetime(df['Date_Occurred'], errors='coerce')
    df['Date_Reported'] = pd.to_datetime(df['Date_Reported'], errors='coerce')
    df['Month_Occurred'] = df['Date_Occurred'].dt.to_period("M").astype(str)
    df['Year'] = df['Date_Occurred'].dt.year
    df['Month'] = df['Date_Occurred'].dt.month
    df['Day'] = df['Date_Occurred'].dt.day
    df['Hour'] = df['Date_Occurred'].dt.hour
    df = df[df['Victim_Age'] >= 0]
    df = df.drop_duplicates()
    df['Weapon_Description'].fillna("UNKNOWN", inplace=True)
    df['Victim_Sex'].fillna("X", inplace=True)
    df['Victim_Descent'].fillna("Unknown", inplace=True)
    df['Status_Description'].fillna("Unknown", inplace=True)
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Options")
categories = st.sidebar.multiselect("Crime Categories", df['Crime_Category'].dropna().unique(), default=df['Crime_Category'].dropna().unique())
weapons = st.sidebar.multiselect("Weapon Used", df['Weapon_Description'].dropna().unique(), default=df['Weapon_Description'].dropna().unique())
sexes = st.sidebar.multiselect("Victim Sex", df['Victim_Sex'].dropna().unique(), default=df['Victim_Sex'].dropna().unique())

min_date = df['Date_Occurred'].min()
max_date = df['Date_Occurred'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Apply filters
filtered_df = df[
    (df['Crime_Category'].isin(categories)) &
    (df['Weapon_Description'].isin(weapons)) &
    (df['Victim_Sex'].isin(sexes)) &
    (df['Date_Occurred'] >= pd.to_datetime(date_range[0])) &
    (df['Date_Occurred'] <= pd.to_datetime(date_range[1]))
]

# Crime Category Distribution
st.subheader("Crime Category Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 4))
train['Crime_Category'].value_counts().plot(kind='pie', autopct='%.2f%%')
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# Victim Age Histogram
st.subheader("Victim Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df['Victim_Age'], bins=30, kde=True, ax=ax2)
st.pyplot(fig2)

# Monthly Trend
st.subheader("Monthly Crime Trend")
monthly = filtered_df.groupby('Month_Occurred').size().reset_index(name='Crimes')
fig3, ax3 = plt.subplots()
sns.lineplot(data=monthly, x='Month_Occurred', y='Crimes', marker="o", ax=ax3)
ax3.tick_params(axis='x', rotation=45)
st.pyplot(fig3)

# Weapon Usage
st.subheader("Top 10 Weapons Used")
top_weapons = filtered_df['Weapon_Description'].value_counts().head(10)
fig4, ax4 = plt.subplots()
sns.barplot(y=top_weapons.index, x=top_weapons.values, ax=ax4)
st.pyplot(fig4)

# Victim Sex Distribution
st.subheader("Victim Sex Distribution")
fig5, ax5 = plt.subplots()
sns.countplot(data=filtered_df, x='Victim_Sex', ax=ax5)
st.pyplot(fig5)

# Age Group vs No. of Reportings
st.subheader("Age Group vs No. of Reportings")
age_bins = pd.cut(filtered_df['Victim_Age'], bins=[0, 5, 17, 30, 60, 99], labels=["Infants", "Children", "Young Adults", "Middle Aged", "Elderly"])
age_counts = age_bins.value_counts().sort_index()
fig6, ax6 = plt.subplots()
sns.barplot(x=age_counts.index, y=age_counts.values, ax=ax6)
ax6.set_xlabel("Age Group")
ax6.set_ylabel("Number of Reportings")
st.pyplot(fig6)

# Victim Descent vs No. of Reportings
st.subheader("Victim Descent vs No. of Reportings")
fig7, ax7 = plt.subplots(figsize=(12, 5))
sns.countplot(data=filtered_df, y='Victim_Descent', order=filtered_df['Victim_Descent'].value_counts().index, ax=ax7)
st.pyplot(fig7)

# Status Description vs No. of Reportings
st.subheader("Status Description vs No. of Reportings")
fig8, ax8 = plt.subplots(figsize=(10, 4))
sns.countplot(data=filtered_df, x='Status_Description', order=filtered_df['Status_Description'].value_counts().index, ax=ax8)
ax8.tick_params(axis='x', rotation=45)
st.pyplot(fig8)

# Crime Category vs Average No. of Reportings
st.subheader("Crime Category vs Average Monthly Reportings")
monthly_crime_avg = filtered_df.groupby(['Crime_Category', 'Month_Occurred']).size().reset_index(name='Count')
monthly_avg = monthly_crime_avg.groupby('Crime_Category')['Count'].mean().sort_values(ascending=False)
fig9, ax9 = plt.subplots(figsize=(10, 5))
sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax9)
ax9.tick_params(axis='x', rotation=45)
ax9.set_ylabel("Average Reportings per Month")
st.pyplot(fig9)

# Time Segments vs No. of Reportings
st.subheader("Time Segments vs No. of Reportings")
def get_time_segment(hour):
    if 400 <= hour <= 759:
        return 'Early Morning'
    elif 800 <= hour <= 1159:
        return 'Morning'
    elif 1200 <= hour <= 1559:
        return 'Afternoon'
    elif 1600 <= hour <= 1959:
        return 'Evening'
    elif 1800 <= hour <= 2359:
        return 'Night'
    elif 1 <= hour <= 359:
        return 'Late Night'
    else:
        return 'Unknown'

filtered_df['Time_Segment'] = filtered_df['Hour'].apply(get_time_segment)
fig10, ax10 = plt.subplots()
sns.countplot(data=filtered_df, x='Time_Segment', order=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'], ax=ax10)
st.pyplot(fig10)

# Weapon Used vs Average No. of Reportings
st.subheader("Weapon Used vs Average Monthly Reportings")
monthly_weapon_avg = filtered_df.groupby(['Weapon_Description', 'Month_Occurred']).size().reset_index(name='Count')
weapon_avg = monthly_weapon_avg.groupby('Weapon_Description')['Count'].mean().sort_values(ascending=False).head(10)
fig11, ax11 = plt.subplots(figsize=(10, 5))
sns.barplot(x=weapon_avg.index, y=weapon_avg.values, ax=ax11)
ax11.tick_params(axis='x', rotation=45)
ax11.set_ylabel("Average Monthly Reportings")
st.pyplot(fig11)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Oorja Gund</p>", unsafe_allow_html=True)
