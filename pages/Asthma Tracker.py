import streamlit as st
import csv
from email.message import EmailMessage
import smtplib
import ssl


st.set_page_config(page_title='Asthma Disease Tracker', layout='wide')
st.title('Asthma Disease Dashboard')

data = {}
severity_percentage = 0

co1,co2,co3 = st.columns(3)
# Define feature weights based on importance values
feature_weights = {
    'Pains': 0.161170,
    'Tiredness': 0.147855,
    'Runny-Nose': 0.146093,
    'Dry-Cough': 0.140978,
    'Nasal-Congestion': 0.140762,
    'Sore-Throat': 0.131730,
    'Difficulty-in-Breathing': 0.131413
}

def email_alert():
    email_sender = "99210041261@klu.ac.in"
    email_password = "wsno odxi crqz gbaa"
    email_recipient = "pmanjuvallabha@gmail.com"
    subject = "Asthma Alert"
    body = "High risk of Asthma attack!"
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_recipient
    em['subject'] = subject
    em.set_content(body)

    contex = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=contex) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_recipient, em.as_string())


# Define a function to calculate severity percentage based on feature weights
def calculate_severity_percentage(symptoms):
    severity_percentage = 0
    for symptom, value in symptoms.items():
        if symptom in feature_weights:
            severity_percentage += value * feature_weights[symptom]
    return severity_percentage * 100  # Multiply by 100 to get percentage


with co1:
    #st.write('Enter the symptoms below:')
        tiredness = st.selectbox('Tiredness', ['<Select>','Yes', 'No'])
        dry_cough = st.selectbox('Dry Cough', ['<Select>','Yes', 'No'])
        difficulty_breathing = st.selectbox('Difficulty in Breathing', ['<Select>','Yes', 'No'])
with co2:
        sore_throat = st.selectbox('Sore Throat', ['<Select>','Yes', 'No'])
        pains = st.selectbox('Pains', ['<Select>','Yes', 'No'])
        nasal_congestion = st.selectbox('Nasal Congestion', ['<Select>','Yes', 'No'])
with co3:
        runny_nose = st.selectbox('Runny Nose', ['<Select>','Yes', 'No'])

user_input = {
        'Tiredness': 1 if tiredness == 'Yes' else 0,
        'Dry-Cough': 1 if dry_cough == 'Yes' else 0,
        'Difficulty-in-Breathing': 1 if difficulty_breathing == 'Yes' else 0,
        'Sore-Throat': 1 if sore_throat == 'Yes' else 0,
        'Pains': 1 if pains == 'Yes' else 0,
        'Nasal-Congestion': 1 if nasal_congestion == 'Yes' else 0,
        'Runny-Nose': 1 if runny_nose == 'Yes' else 0,
    }
with co1:
    button  = st.button('Calculate Severity')
    with co2:
        alert = st.button('Send Alert')
    if button:
        severity_percentage = calculate_severity_percentage(user_input)
        if severity_percentage > 50:
            st.error('High risk of Asthma attack!')

if alert:
            try:
                    email_alert()
                    st.success('Alert sent to the registered email!')
            except Exception as e:
                    st.error(f"Failed to send email: {e}")
    


# Read existing data from the CSV file, if it exists
try:
    with open('user_data.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            username = row['Username']
            date = row['Date']
            severity = row['Severity']
            if username in data:
                data[username].append({'date': date, 'severity': severity})
            else:
                data[username] = [{'date': date, 'severity': severity}]
except FileNotFoundError:
    pass

with st.sidebar:
# Get user input
    username = st.text_input("Enter the username: ")
    date = st.date_input("Enter the date:")



# Check if user data already exists
if username in data:
    # Update existing user data
    data[username].append({'date': date, 'severity': severity_percentage})
else:
    # Add new user data
    data[username] = [{'date': date, 'severity': severity_percentage}]


with st.sidebar:
# Save data to CSV file
    if button:
        with open('user_data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Username', 'Date', 'Severity'])  # Write header
            for username, entries in data.items():
                for entry in entries:
                    writer.writerow([username, entry['date'], entry['severity']])

