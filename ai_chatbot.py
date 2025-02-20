# libraries
import os
from datetime import datetime
from pytz import timezone
from random import randint
import plotly.io as pio
import plotly.offline as pyo
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from folium.plugins import AntPath
import folium
import matplotlib.dates as mdates
from datetime import timedelta
from io import BytesIO
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz
import pandas as pd
import re
import spacy
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import difflib
from nltk.corpus import stopwords
import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv, find_dotenv

load_dotenv()

path = os.path.abspath(__file__ + "/..") + "/"
api_url = os.environ["FLASK_URL"]
lemmatizer = WordNetLemmatizer()
import string
import math

stop_words = set(stopwords.words('english'))
ignore_words = ['?', '!', '.', ',']

# chat initialization
model = load_model(path + "chatbot_model.h5")
intents = json.loads(open(path + "intents.json").read())
words = pickle.load(open(path + "words.pkl", "rb"))
classes = pickle.load(open(path + "classes.pkl", "rb"))

df = pd.read_excel(path + 'Shipment_Data.xlsx')
df['Mode'] = df['Mode'].str.lower()
df['Actual Revenue'] = df['Actual Revenue'].fillna(0)
df['Booked Date'] = df['Booked Date'].dt.strftime('%m/%d/%Y')
revenue_data = df.to_dict(orient='records')

# Import the cp wise data
company_df = pd.read_excel(path + "CP wise Stats _FTL_Leader_KPI_Dashboard.xlsx")
company_df = company_df.fillna(0)
company_copy_df = company_df.copy()

# Lanes data import
lane_df = pd.read_excel(path + 'shipment_lanes_data.xlsx',
                        sheet_name='PRISM Shipment Data')
filtered_df = lane_df[['state_lane', 'txn_count', 'estimated_revenue']]
lane_data = filtered_df.groupby("state_lane").agg(
    {"txn_count": "sum", "estimated_revenue": "sum"}).reset_index()
sorted_df = lane_data.sort_values("txn_count", ascending=False)

# import the datewise cp data
data_df = pd.read_csv(path + 'quotes_data_ftl_leader_kpi_dashboard.csv')
data_df["CP Name"] = data_df["CP Name"].str.lower()
data_df = data_df.fillna(0)
# Assuming data_df is your DataFrame containing the "Quote CreatedAt" column
data_df["Quote CreatedAt"] = pd.to_datetime(data_df["Quote CreatedAt"])

# Now you can use the .dt accessor to format the date
data_df['Booked Date'] = data_df['Quote CreatedAt'].dt.strftime('%m/%d/%Y')

nlp_ner = spacy.load(path + "model-best")
word_mapping_month = {
    'January': ['jan', 'january'],
    'February': ['february', 'feb'],
    'March': ['march', 'mar'],
    'April': ['april', 'apr'],
    'June': ['june', 'jun'],
    'July': ['july', 'jul'],
    'August': ['august', 'aug'],
    'September': ['september', 'sep'],
    'October': ['october', 'oct'],
    'November': ['november', 'nov'],
    'December': ['december', 'dec'],
    'revenue': ['revenue', 'revnue']}

word_mapping_day = {'1st': ['1', '1st', 'first'],
                    '2nd': ['2', '2nd', 'second'],
                    '3rd': ['3', '3rd', 'third'],
                    '4th': ['4', '4th', 'fourth'],
                    '5th': ['5', '5th', 'fifth'],
                    '6th': ['6', '6th', 'sixth'],
                    '7th': ['7', '7th', 'seventh'],
                    '8th': ['8', '8th', 'eighth'],
                    '9th': ['9', '9th', 'ninth'],
                    '10th': ['10', '10th', 'tenth'],
                    '11th': ['11', '11th', 'eleventh'],
                    '12th': ['12', '12th', 'twelfth'],
                    '13th': ['13', '13th', 'thirteenth'],
                    '14th': ['14', '14th', 'fourteenth'],
                    '15th': ['15', '15th', 'fifteenth'],
                    '16th': ['16', '16th', 'sixteenth'],
                    '17th': ['17', '17th', 'seventeenth'],
                    '18th': ['18', '18th', 'eighteenth'],
                    '19th': ['19', '19th', 'nineteenth'],
                    '20th': ['20', '20th', 'twentieth'],
                    '21st': ['21', '21st', 'twenty-first'],
                    '22nd': ['22', '22nd', 'twenty-second'],
                    '23rd': ['23', '23rd', 'twenty-third'],
                    '24th': ['24', '24th', 'twenty-fourth'],
                    '25th': ['25', '25th', 'twenty-fifth'],
                    '26th': ['26', '26th', 'twenty-sixth'],
                    '27th': ['27', '27th', 'twenty-seventh'],
                    '28th': ['28', '28th', 'twenty-eighth'],
                    '29th': ['29', '29th', 'twenty-ninth'],
                    '30th': ['30', '30th', 'thirtieth'],
                    '31st': ['31', '31st', 'thirty-first']
                    }

word_corpus = words


def find_closest_match(user_input):
    closest_match = difflib.get_close_matches(
        user_input, word_corpus, n=1, cutoff=0.85)
    print("close>>>", closest_match, user_input)
    if len(closest_match) > 0:
        return closest_match[0]
    else:
        return None


def clean_up_sentence(sentence):
    cp_name = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence_words = nltk.word_tokenize(cp_name)
    lemma_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words if w not in ignore_words]
    print("lemma>>", lemma_words)
    # filtered_words = [
    #     word for word in lemma_words if not word.lower() in lemma_words]
    corrected_sentence = [find_closest_match(word) for word in lemma_words]
    print(corrected_sentence)
    corpus_plus_words = [
        word for word in corrected_sentence if word in word_corpus]
    # sentence_words = [lemmatizer.lemmatize(word) for word in corpus_plus_words]
    return corpus_plus_words


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


context = {}


def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def replace_month_words_in_sentence(sentence, word_mapping):
    words = re.findall(r'\w+|[^\w\s]', sentence)
    result = []
    for word in words:
        for key, values in word_mapping.items():
            for value in values:
                similarity = fuzz.ratio(word.lower(), value.lower())
                if similarity >= 90:
                    word = key
                    break
        result.append(word)
    return ' '.join(result)


def replace_day_words_in_sentence(sentence, word_mapping):
    words = re.findall(r'\w+|[^\w\s]', sentence)
    result = []
    for word in words:
        for key, values in word_mapping.items():
            for value in values:
                similarity = fuzz.ratio(word.lower(), value.lower())
                if similarity >= 90:
                    word = key
                    break
        result.append(word)
    return ' '.join(result)


def convert_date(date_expression):
    if date_expression[1] == 'DATE':
        try:
            date_str = date_expression[0]
            ordinal_suffixes = {'st': 'th', 'nd': 'th', 'rd': 'th'}

            # Handle the case where the day and month are reversed
            if date_str.count(' ') == 1:
                date_parts = date_str.split()
                day = date_parts[0]
                month = date_parts[1]

                for suffix, replacement in ordinal_suffixes.items():
                    day = day.replace(suffix, replacement)
                date_str = ' '.join([day, month])

            for suffix, replacement in ordinal_suffixes.items():
                date_str = date_str.replace(suffix, replacement)

            try:
                date = datetime.strptime(date_str, '%dth %B')
            except:
                date = datetime.strptime(date_str, '%B %dth')

            converted_date = date.replace(year=datetime.now().year)

            if converted_date > datetime.now():
                converted_date = converted_date.replace(
                    year=datetime.now().year - 1)

            return converted_date.strftime("%m/%d/%Y")
        except ValueError:
            return None  # Return None for invalid dates
    else:
        return None  # Return None for invalid expressions


def calculate_revenue(modes, start_date, end_date):
    # Define custom colors for each mode
    color_palette = {
        'fcl': '#AA3939',  # Dark red
        'ftl': '#ffa600',  # Orange
        'lcl': '#665191',  # Purple
        'ltl': '#ff7c43',  # Coral
        'parcel': '#bc5090',  # Magenta
        'customs': '#7a5195',  # Indigo
        'drayage': '#FFD700',  # Gold
        'air': '#58508d',  # Violet
        'others': '#40a798',  # Teal
        'total': '#17becf'  # Light blue
    }

    # Initialize dictionaries to store mode-wise data
    mode_revenues = {mode: [] for mode in modes}
    mode_dates = {mode: [] for mode in modes}

    # Convert start_date and end_date strings to datetime objects
    start_date = datetime.strptime(start_date, "%m/%d/%Y")
    end_date = datetime.strptime(end_date, "%m/%d/%Y")

    all_mode_total_revenue = 0.0
    date_format = "%m/%d/%Y"
    for entry in revenue_data:
        booked_date = datetime.strptime(entry['Booked Date'], date_format)
        if start_date <= booked_date <= end_date:
            all_mode_total_revenue += entry['Actual Revenue']

    # Calculate the number of days between start_date and end_date
    num_days = (end_date - start_date).days

    # Iterate through revenue data and calculate the total revenue for each mode on a daily basis
    total_revenue = 0  # Initialize total revenue
    for i in range(num_days + 1):
        current_date = start_date + timedelta(days=i)

        # Iterate through revenue data for the current date
        for entry in revenue_data:
            entry_date = datetime.strptime(entry['Booked Date'], "%m/%d/%Y")
            lowercase_modes = list(map(str.lower, modes))
            if entry_date == current_date and entry["Mode"].lower() in lowercase_modes:
                mode = entry["Mode"]
                mode_revenues[mode].append(entry.get('Actual Revenue', 0))
                mode_dates[mode].append(current_date)
                total_revenue += entry.get('Actual Revenue', 0)

    # Create mode-wise line graphs using Matplotlib
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    colors = [color_palette[mode] for mode in modes]

    # Plot revenue trend by mode
    for i, mode in enumerate(modes):
        mode_dates_mpl = mdates.date2num(mode_dates[mode])

        # Smooth the revenue data using interpolation
        mode_revenues_interp = np.interp(
            mode_dates_mpl, mode_dates_mpl, mode_revenues[mode])

        ax1.plot(mode_dates_mpl, mode_revenues_interp,
                 label=mode, color=colors[i])

    ax1.set_ylabel('Revenue')
    ax1.set_title('Revenue Trend (Mode-wise - Daily)')
    ax1.legend()

    # Format x-axis tick labels with day and month only
    date_labels = [date.strftime("%m/%d") for date in mode_dates[modes[0]]]
    # Format x-axis tick labels with month and year
    ax1.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: "${:,.2f}".format(x)))
    # Show tick labels for every 7 days
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(
        '%m/%d'))  # Format tick labels as month/day
    ax1.tick_params(axis='x', rotation=90)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # Calculate total revenue by mode
    mode_total_revenue = {mode: sum(mode_revenues[mode]) for mode in modes}

    other_modes_revenue = all_mode_total_revenue - \
                          sum(mode_total_revenue.values())
    mode_total_revenue['others'] = other_modes_revenue

    print("keys", mode_total_revenue.keys())

    # Sort the mode_total_revenue dictionary in descending order by value
    sorted_mode_total_revenue = dict(
        sorted(mode_total_revenue.items(), key=lambda x: x[1], reverse=True))

    # Calculate revenue percentage for each mode
    mode_percentages = {mode: (revenue / total_revenue) *
                              100 for mode, revenue in sorted_mode_total_revenue.items()}

    # Plot revenue by mode as a pie chart
    wedges, texts, autotexts = ax2.pie(sorted_mode_total_revenue.values(), labels=sorted_mode_total_revenue.keys(),
                                       colors=[color_palette[mode] for mode in sorted_mode_total_revenue.keys()],
                                       autopct='%1.1f%%', labeldistance=1.1)

    ax2.set_title('Percentage Share of Revenue')

    mode_total_revenue['total'] = all_mode_total_revenue
    # Sort the mode_total_revenue dictionary in descending order by value
    sorted_mode_total_revenue = dict(
        sorted(mode_total_revenue.items(), key=lambda x: x[1], reverse=True))

    # Plot revenue in dollars as a horizontal bar graph
    modes_list = list(sorted_mode_total_revenue.keys())

    ax3.barh(modes_list, sorted_mode_total_revenue.values(),
             color=[color_palette[mode] for mode in modes_list])
    ax3.set_xlabel('Revenue ($)')
    ax3.set_ylabel('Mode')
    ax3.set_title('Revenue by Mode (Descending Order)')
    ax3.invert_yaxis()
    # Remove spines (borders) of the graph
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)

    # Hide y-axis ticks and tick labels
    ax3.yaxis.set_ticks_position('none')

    # Set x-axis tick format to display as dollars
    # ax3.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,.2f}".format(x)))

    # Add revenue amount on the right side of each bar
    for y, x in enumerate(sorted_mode_total_revenue.values()):
        ax3.annotate(f"${x:.2f}", (x + 10, y), va='center')

    plt.tight_layout()
    # plt.savefig('revenue_graph.png')
    # plt.close()

    # Save the graphs as images in memory
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=60)
    plt.close()

    # Convert the image buffer to base64 encoded string
    img_str = base64.b64encode(img_buffer.getvalue()).decode()

    # Generate the HTML output
    output = f"From {start_date.date().isoformat()} to {end_date.date().isoformat()}:\n"

    # Generate the revenue table in HTML format
    table_html = '<table style="border-collapse: collapse; border: 1px solid black;">'
    table_html += '<tr><th style="border: 1px solid black;">Mode</th><th style="border: 1px solid black;">Revenue</th></tr>'

    for mode, revenue in sorted_mode_total_revenue.items():
        revenue_formatted = "{:.2f}".format(revenue)
        table_html += f'<tr style="border: 1px solid black;"><td style="border: 1px solid black;">{mode.upper()}</td><td style="border: 1px solid black;">${revenue_formatted}</td></tr>'

    table_html += '</table>'

    # Embed the image in the HTML output
    img_html = f'<img src="data:image/png;base64,{img_str}">'

    output += img_html
    output += table_html

    return output


# Top and bottom lanes

def get_top_bottom_lanes(msg):
    doc = nlp_ner(msg)  # input sample text
    ents = [(e.text, e.label_) for e in doc.ents]
    ner_dict = {label: value for value, label in ents}
    print(ner_dict)

    if 'LANES' in ner_dict and ner_dict['LANES'].startswith('top'):
        n = int(ner_dict['LANES'].split()[1])
        lanes = sorted_df.head(n)
        lanes_text = f"top {n} lanes"
        print("Lanes>>>>", lanes)
    else:
        n = int(ner_dict['LANES'].split()[1])
        lanes = sorted_df.tail(n)
        lanes_text = f"bottom {n} lanes"

    # Generate the table HTML
    table_html = '<table style="border-collapse: collapse; border: 1px solid black;">'
    table_html += '<tr><th style="border: 1px solid black;">State Lane</th><th style="border: 1px solid black;">Transaction Count</th><th style="border: 1px solid black;">Estimated Revenue</th></tr>'

    # Add rows to the table
    for _, row in lanes.iterrows():
        table_html += f'<tr style="border: 1px solid black;"><td style="border: 1px solid black;">{row["state_lane"]}</td><td style="border: 1px solid black;">{row["txn_count"]}</td><td style="border: 1px solid black;">${row["estimated_revenue"]:,.2f}</td></tr>'

    table_html += '</table>'

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot transaction count
    ax1.bar(lanes["state_lane"], lanes["txn_count"], color='gray')
    ax1.set_xlabel("State Lane")
    ax1.set_ylabel("Transaction Count")
    ax1.set_title(f"Transaction Count by State Lane ({lanes_text})")

    # Add number labels above the bars
    for i, (_, row) in enumerate(lanes.iterrows()):
        ax1.text(i, row["txn_count"], str(
            row["txn_count"]), ha='center', va='bottom')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)

    # Plot estimated revenue
    ax2.bar(lanes["state_lane"], lanes["estimated_revenue"], color='#5fdae3')
    ax2.set_xlabel("State Lane")
    ax2.set_ylabel("Estimated Revenue ($)")
    ax2.set_title(f"Estimated Revenue by State Lane ({lanes_text})")

    # Add number labels above the bars
    for i, (_, row) in enumerate(lanes.iterrows()):
        ax2.text(i, row["estimated_revenue"],
                 f'${row["estimated_revenue"]:,.2f}', ha='center', va='bottom')

    # Rotate x-axis labels vertically
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.set_xticklabels(lanes["state_lane"], rotation='vertical')
    ax2.set_xticklabels(lanes["state_lane"], rotation='vertical')
    plt.tight_layout()

    # Save the graph as an image in memory
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=60)
    plt.close()

    # Convert the image buffer to a base64 encoded string
    img_str = base64.b64encode(img_buffer.getvalue()).decode()

    # Generate the HTML for the image
    img_html = f'<img src="data:image/png;base64,{img_str}">'

    # Create the maps
    # Create a geocoder object using Nominatim
    user_agent = 'user_me_{}'.format(randint(10000, 99999))
    print(user_agent)
    geolocator = Nominatim(user_agent=user_agent, timeout=5)
    # Extract start and end locations from the state_lane column
    lanes['start_location'] = lanes['state_lane'].str.split(' to ').str[0]
    lanes['end_location'] = lanes['state_lane'].str.split(' to ').str[1]

    # Create a map centered on the first start location
    first_start_location = lanes['start_location'].iloc[0]
    first_start_geocode = geolocator.geocode(first_start_location)
    map = folium.Map(location=[first_start_geocode.latitude,
                               first_start_geocode.longitude], zoom_start=2)

    folium.TileLayer('Stamen Terrain').add_to(map)
    folium.TileLayer('Stamen Toner').add_to(map)
    folium.TileLayer('Stamen Water Color').add_to(map)
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('cartodbdark_matter').add_to(map)
    folium.LayerControl().add_to(map)

    # Add flight symbols and location names for each location pair
    for index, row in lanes.iterrows():
        start_location_name = row['start_location']
        end_location_name = row['end_location']

        # Geocode start and end locations
        start_location = geolocator.geocode(start_location_name)
        end_location = geolocator.geocode(end_location_name)

        if start_location is not None and end_location is not None:
            start_coords = [start_location.latitude, start_location.longitude]
            end_coords = [end_location.latitude, end_location.longitude]

            # Add flight takeoff symbol at the start location
            takeoff_icon = folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html='<i class="fas fa-plane" style="font-size: 20px; color: red; transform: rotate(180deg);"></i>'
            )
            takeoff_marker = folium.Marker(
                location=start_coords,
                icon=takeoff_icon
            )
            takeoff_marker.add_to(map)

            # Add flight landing symbol at the end location
            landing_icon = folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html='<i class="fas fa-plane" style="font-size: 20px; color: green; transform: rotate(180deg);"></i>'
            )
            landing_marker = folium.Marker(
                location=end_coords,
                icon=landing_icon
            )
            landing_marker.add_to(map)

            # Add location names as popups
            start_popup = folium.Popup(f'Start: {start_location_name}')
            end_popup = folium.Popup(f'End: {end_location_name}')
            folium.Marker(
                location=start_coords,
                popup=start_popup,
                icon=folium.Icon(icon='cloud')
            ).add_to(map)
            folium.Marker(
                location=end_coords,
                popup=end_popup,
                icon=folium.Icon(icon='cloud')
            ).add_to(map)

            # Create a curved dotted line between the start and end locations
            line = AntPath(
                locations=[start_coords, end_coords],
                dash_array=[10, 20],
                weight=2,
                color='blue'
            )
            line.add_to(map)

    # Add a legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 1px solid gray;">
        <div><i class="fas fa-plane" style="font-size: 20px; color: red;"></i> Origin</div>
        <div><i class="fas fa-plane" style="font-size: 20px; color: green;"></i> Destination</div>
    </div>
    '''
    map.get_root().html.add_child(folium.Element(legend_html))

    # Save the map as an HTML file
    file_name_time_format = "%Y_%m_%d_%H_%M_%S"
    current_time = datetime.now(timezone("UTC")).strftime(file_name_time_format)
    # file_path = path + '../../frontend/public/data/'
    file_path = "templates/"
    file_name = 'map' + current_time + '.html'
    map.save(file_path + file_name)
    # h = HTMLParser()
    # map_html = h.unescape(map._repr_html_())
    map_html = "[View Lanes](" + api_url + "html_page?page_name=" + file_name + ")"

    # Concatenate the table HTML and image HTML
    output = map_html + img_html + table_html
    return output


def generate_result(input_text):
    # print("+"*30)
    # input_text = input("User>>")
    input_text = (input_text).lower()
    # print("+"*30)
    input_text = replace_month_words_in_sentence(
        input_text, word_mapping_month)
    doc = nlp_ner(input_text)  # input sample text
    ents = [(e.text, e.label_) for e in doc.ents]
    ner_dict = {label: value for value, label in ents}
    print(ner_dict)

    if 'DAYS' in ner_dict:
        days_value = ner_dict['DAYS'].split()[0]
        if days_value.isdigit():
            duration = int(days_value)
            start_date = (datetime.today() -
                          timedelta(days=duration)).strftime("%m/%d/%Y")
            end_date = datetime.today().strftime("%m/%d/%Y")
            modes = [item[0] for item in ents if item[1] == 'MODE']
            # print(start_date,end_date,modes)
            revenue = calculate_revenue(modes, start_date, end_date)
            return revenue
        else:
            print("Invalid days value.")
            exit()

    elif 'MONTH' in ner_dict and 'YEAR' in ner_dict and 'WEEK' in ner_dict:
        month_value = ner_dict['MONTH']
        year_value = ner_dict['YEAR']
        week_value = ner_dict['WEEK']

        # Extract the numeric value from the week string
        numeric_week = int(re.search(r'\d+', week_value).group())

        # Find the start date and end date based on month, year, and week values
        start_date = datetime.strptime(f'{month_value} {year_value}', '%B %Y')
        start_date += timedelta(weeks=numeric_week - 1)
        end_date = start_date + timedelta(weeks=1) - timedelta(days=1)

        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")

        modes = [item[0] for item in ents if item[1] == 'MODE']
        revenue = calculate_revenue(modes, start_date_str, end_date_str)
        return revenue

    elif 'MONTH' in ner_dict and 'YEAR' not in ner_dict:
        # Get the month from the dictionary
        month = ner_dict['MONTH']

        # Get the current year and month
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month

        # Get the month number based on the month name
        month_number = list(calendar.month_name).index(month)

        # Assign the year based on the month comparison
        if month_number > current_month:
            previous_year = current_year - 1
        else:
            previous_year = current_year

        # Calculate the start and end dates
        start_date = datetime(previous_year, month_number, 1)
        end_date = (
                start_date + timedelta(days=calendar.monthrange(previous_year, month_number)[1] - 1))

        # Format the dates as strings in the desired format
        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")

        modes = [item[0] for item in ents if item[1] == 'MODE']
        revenue = calculate_revenue(modes, start_date_str, end_date_str)
        return revenue

    elif 'MONTH' in ner_dict and 'YEAR' in ner_dict:
        print("YOHO")
        # Get the month from the dictionary
        month = ner_dict['MONTH']

        # Get the current year and month
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month

        # Get the month number based on the month name
        month_number = list(calendar.month_name).index(month)
        year = int(ner_dict['YEAR'])

        # Calculate the start and end dates
        start_date = datetime(year, month_number, 1)
        end_date = (
                start_date + timedelta(days=calendar.monthrange(year, month_number)[1] - 1))

        # Format the dates as strings in the desired format
        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")

        modes = [item[0] for item in ents if item[1] == 'MODE']
        revenue = calculate_revenue(modes, start_date_str, end_date_str)
        return revenue

    elif 'YEAR' in ner_dict and 'QUARTER' in ner_dict:
        year = int(ner_dict['YEAR'])
        quarter_str = ner_dict['QUARTER']

        # Extract the quarter number using regular expressions
        quarter_match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\b', quarter_str)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            start_month = (quarter - 1) * 3 + 1
            start_date = datetime(year, start_month, 1)
            end_date = start_date + timedelta(days=89)

            # Format the dates as strings in the desired format
            start_date_str = start_date.strftime("%m/%d/%Y")
            end_date_str = end_date.strftime("%m/%d/%Y")

            modes = [item[0] for item in ents if item[1] == 'MODE']
            revenue = calculate_revenue(modes, start_date_str, end_date_str)
            return revenue
        else:
            print("Invalid quarter format")
    else:
        input_text = replace_month_words_in_sentence(
            input_text, word_mapping_month)
        input_text = replace_day_words_in_sentence(
            input_text, word_mapping_day)
        # print("Changed_data-----",input_text)
        doc = nlp_ner(input_text)  # input sample text
        ents = [(e.text, e.label_) for e in doc.ents]
        converted_dates = []
        for date_expression in ents:
            converted_date = convert_date(date_expression)
            if converted_date is not None:
                converted_dates.append(converted_date)

        # Convert the date strings to datetime objects
        dates = [datetime.strptime(date, '%m/%d/%Y')
                 for date in converted_dates]

        # Find the maximum and minimum dates
        max_date = max(dates)
        min_date = min(dates)

        # Convert the max_date and min_date back to string format
        max_date_str = max_date.strftime('%m/%d/%Y')
        min_date_str = min_date.strftime('%m/%d/%Y')

        end_date = max_date_str
        start_date = min_date_str
        # convert the list to a dictionary
        ner_dict = {label: value for value, label in ents}
        # print(ner_dict)
        modes = [item[0] for item in ents if item[1] == 'MODE']
        # print(start_date,end_date,modes)

        revenue = calculate_revenue(modes, start_date, end_date)

        return revenue


#####generate the cp_wise quote request


def filter_data(start_date, end_date, company_name, query):
    # Load the data into a DataFrame
    df = data_df  # Replace 'data.csv' with the actual file path or DataFrame

    # Convert the date columns to datetime format
    df['Quote CreatedAt'] = pd.to_datetime(df['Quote CreatedAt'])
    df['Booked Date'] = pd.to_datetime(df['Booked Date'], format='%m/%d/%Y')

    # Define the filter conditions
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')

    # Filter the DataFrame based on the conditions
    filtered_df = df[
        (df['CP Name'].str.lower() == company_name.lower()) &
        (df['Booked Date'].dt.date >= start_date.date()) &
        (df['Booked Date'].dt.date <= end_date.date())
        ]
    
    filtered_df.rename(columns={'Avg. Booking Lead Time (In Minutes)': 'Booking Lead Time'}, inplace=True)
    # Get the numeric columns
    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns

    # Add mean, average, and standard deviation columns to the filtered DataFrame
    for column in numeric_columns:
        filtered_df[f"Mean {column}"] = filtered_df[column]
        filtered_df[f"Standard Deviation {column}"] = filtered_df[column]
        filtered_df[f"Average {column}"] = filtered_df[column]

    # Reset the index of the DataFrame
    filtered_df.reset_index(drop=True, inplace=True)

    # If there is a query, filter the DataFrame based on similarity
    if query:
        metrics_list = list(filtered_df.columns)

        similar_sentences = find_similar_sentence(
            query, metrics_list, threshold=0.85)
        if similar_sentences:
            columns_to_filter = [
                sentence for sentence in similar_sentences if sentence in filtered_df.columns]
            columns_to_filter.append("Booked Date")
            if columns_to_filter:
                filtered_df = filtered_df[columns_to_filter]


    filtered_df['Month'] = filtered_df['Booked Date'].dt.to_period('M')
    # Select only the numeric columns for summation
    numeric_columns = filtered_df.select_dtypes(include='number').columns
    # Group by 'Month' and sum the numeric columns
    # grouped_data = filtered_df.groupby('Month')[numeric_columns].sum().reset_index()

    if any(column in numeric_columns for column in ['Total Quotes Requested', 'Total Requested Shipment',
                                                    'Total Quotes Awarded', 'Total Booked Shipment',
                                                    'Total Shipment Failed by CP', 'Total Quotes Failed by Vendor',
                                                    'Abandon Shipments by CP', 'Abandon Shipments by Vendor',
                                                    'Booking Lead Time']):
        # Group by 'Month' and sum the numeric columns
        grouped_data = filtered_df.groupby('Month')[numeric_columns].sum().reset_index()
        total_sum = grouped_data[numeric_columns].sum()

    elif any(column in numeric_columns for column in ['Standard Deviation Total Quotes Requested',
                                                  'Standard Deviation Total Requested Shipment',
                                                  'Standard Deviation Total Quotes Awarded',
                                                  'Standard Deviation Total Booked Shipment',
                                                  'Standard Deviation Total Shipment Failed by CP',
                                                  'Standard Deviation Total Quotes Failed by Vendor',
                                                  'Standard Deviation Abandon Shipments by CP',
                                                  'Standard Deviation Abandon Shipments by Vendor',
                                                  'Standard Deviation Booking Lead Time']):
    # Group by 'Month' and calculate the standard deviation of numeric columns
        grouped_data = filtered_df.groupby('Month')[numeric_columns].sum().reset_index()
        print("*****************************************",grouped_data)
        total_sum = grouped_data[numeric_columns].std()

    else:
        # Group by 'Month' and calculate the average of numeric columns
        grouped_data = filtered_df.groupby('Month')[numeric_columns].mean().reset_index()
        total_sum = grouped_data[numeric_columns].mean()


    total_sum = total_sum[:9]

    # print("grouped_data",grouped_data)

    # Select the first 10 columns of the grouped_data DataFrame
    grouped_data = grouped_data.iloc[:, :10]

    # Create the HTML table
    html_table = grouped_data.reset_index().to_html(index=False, justify='center', classes='data')

    # Add CSS styling to center the table data
    html_table = html_table.replace('<table', '<table style="border-collapse: collapse; border: 1px solid black;"')

    result_string = f"Yes! For {company_name.upper()} the data from {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')} is as follows:<br>"
    result_string += "<ul>"

    for column, value in total_sum.items():
        result_string += f"<li>The {column} from {start_date.strftime('%m/%d/%Y')} to {end_date.strftime('%m/%d/%Y')} is {value}.</li>"

    result_string += "</ul>"

    # Properly indent the HTML table
    indented_html_table = ""
    for line in html_table.split('\n'):
        indented_html_table += "    " + line + "\n"

    # Combine the result string and indented HTML table
    result_string += f"\nMonthwise Statestics:\n{indented_html_table}"

    # Convert the 'Month' column to string format
    grouped_data['Month'] = grouped_data['Month'].astype(str)

    # Determine the subplot layout
    num_plots = len(grouped_data.columns) - 1  # Exclude the 'Month' column
    num_rows = math.ceil(num_plots / 3)  # Calculate the number of rows

    # Adjust subplot layout for single column case
    if num_plots == 1:
        num_rows = 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    fig.tight_layout(pad=5)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    # Create line chart for each column
    colors = ['magenta', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'red',
              'purple', 'cyan', 'green', 'magenta']  # Define colors for each graph
    for i, column in enumerate(grouped_data.columns[1:]):
        if i >= num_plots:
            break

        ax = axes[i]
        ax.plot(grouped_data['Month'], grouped_data[column], marker='o', color=colors[i])
        ax.set_xlabel('Month')
        ax.set_ylabel(column)
        ax.set_title(f"{column} Trend")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        ax.tick_params(axis='x', rotation=90)

        # Add value labels beside each marker
        for j, value in enumerate(grouped_data[column]):
            ax.annotate(int(value), (grouped_data['Month'][j], value), xytext=(5, 5),
                        textcoords='offset points', ha='left')

    # Remove blank subplots
    if num_plots < len(axes):
        for ax in axes[num_plots:]:
            ax.remove()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot to a file or display it
    # Save the graphs as images in memory
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=85)
    plt.close()
    # Convert the image buffer to a base64 encoded string
    img_str = base64.b64encode(img_buffer.getvalue()).decode()

    # Generate the HTML for the image
    img_html = f'<img src="data:image/png;base64,{img_str}">'

    return img_html + result_string


def find_similar_sentence(query, sentences, threshold):
    # Preprocess the sentences
    sentences_preprocessed = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

    # Preprocess the query
    query_tokens = nltk.word_tokenize(query.lower())

    # Create TF-IDF vectors for the sentences and query
    vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    sentences_vectors = vectorizer.fit_transform(sentences_preprocessed)
    query_vector = vectorizer.transform([query_tokens])

    # Calculate cosine similarity between query and sentences
    similarities = cosine_similarity(query_vector, sentences_vectors)

    # Find sentences above the threshold
    similar_sentences = []
    for index, similarity in enumerate(similarities[0]):
        if similarity > threshold:
            similar_sentences.append(sentences[index])

    return similar_sentences


def generate_result_cpwise(company_name, input_text, saved_msg):
    # print("+"*30)
    # input_text = input("User>>")
    input_text = (input_text).lower()
    # print("+"*30)
    input_text = replace_month_words_in_sentence(
        input_text, word_mapping_month)
    doc = nlp_ner(input_text)  # input sample text
    ents = [(e.text, e.label_) for e in doc.ents]
    ner_dict = {label: value for value, label in ents}
    print(ner_dict)

    if 'DAYS' in ner_dict:
        days_value = ner_dict['DAYS'].split()[0]
        if days_value.isdigit():
            duration = int(days_value)
            start_date = (datetime.today() -
                          timedelta(days=duration)).strftime("%m/%d/%Y")
            end_date = datetime.today().strftime("%m/%d/%Y")
            # print(start_date,end_date,modes)
            revenue = filter_data(start_date, end_date, company_name, saved_msg)
            return revenue
        else:
            print("Invalid days value.")
            exit()

    elif 'MONTH' in ner_dict and 'YEAR' in ner_dict and 'WEEK' in ner_dict:
        month_value = ner_dict['MONTH']
        year_value = ner_dict['YEAR']
        week_value = ner_dict['WEEK']

        # Extract the numeric value from the week string
        numeric_week = int(re.search(r'\d+', week_value).group())

        # Find the start date and end date based on month, year, and week values
        start_date = datetime.strptime(f'{month_value} {year_value}', '%B %Y')
        start_date += timedelta(weeks=numeric_week - 1)
        end_date = start_date + timedelta(weeks=1) - timedelta(days=1)

        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")
        revenue = filter_data(start_date_str, end_date_str, company_name, saved_msg)
        return revenue

    elif 'MONTH' in ner_dict and 'YEAR' not in ner_dict:
        # Get the month from the dictionary
        month = ner_dict['MONTH']

        # Get the current year and month
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month

        # Get the month number based on the month name
        month_number = list(calendar.month_name).index(month)

        # Assign the year based on the month comparison
        if month_number > current_month:
            previous_year = current_year - 1
        else:
            previous_year = current_year

        # Calculate the start and end dates
        start_date = datetime(previous_year, month_number, 1)
        end_date = (
                start_date + timedelta(days=calendar.monthrange(previous_year, month_number)[1] - 1))

        # Format the dates as strings in the desired format
        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")
        revenue = filter_data(start_date_str, end_date_str, company_name, saved_msg)
        return revenue

    elif 'MONTH' in ner_dict and 'YEAR' in ner_dict:
        print("YOHO")
        # Get the month from the dictionary
        month = ner_dict['MONTH']

        # Get the current year and month
        current_date = datetime.now().date()
        current_year = current_date.year
        current_month = current_date.month

        # Get the month number based on the month name
        month_number = list(calendar.month_name).index(month)
        year = int(ner_dict['YEAR'])

        # Calculate the start and end dates
        start_date = datetime(year, month_number, 1)
        end_date = (
                start_date + timedelta(days=calendar.monthrange(year, month_number)[1] - 1))

        # Format the dates as strings in the desired format
        start_date_str = start_date.strftime("%m/%d/%Y")
        end_date_str = end_date.strftime("%m/%d/%Y")
        revenue = filter_data(start_date_str, end_date_str, company_name, saved_msg)
        return revenue

    elif 'YEAR' in ner_dict and 'QUARTER' in ner_dict:
        year = int(ner_dict['YEAR'])
        quarter_str = ner_dict['QUARTER']

        # Extract the quarter number using regular expressions
        quarter_match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\b', quarter_str)
        if quarter_match:
            quarter = int(quarter_match.group(1))
            start_month = (quarter - 1) * 3 + 1
            start_date = datetime(year, start_month, 1)
            end_date = start_date + timedelta(days=89)

            # Format the dates as strings in the desired format
            start_date_str = start_date.strftime("%m/%d/%Y")
            end_date_str = end_date.strftime("%m/%d/%Y")
            revenue = filter_data(start_date_str, end_date_str, company_name, saved_msg)
            return revenue
        else:
            print("Invalid quarter format")
    else:
        input_text = replace_month_words_in_sentence(
            input_text, word_mapping_month)
        input_text = replace_day_words_in_sentence(
            input_text, word_mapping_day)
        # print("Changed_data-----",input_text)
        doc = nlp_ner(input_text)  # input sample text
        ents = [(e.text, e.label_) for e in doc.ents]
        converted_dates = []
        for date_expression in ents:
            converted_date = convert_date(date_expression)
            if converted_date is not None:
                converted_dates.append(converted_date)

        # Convert the date strings to datetime objects
        dates = [datetime.strptime(date, '%m/%d/%Y')
                 for date in converted_dates]

        # Find the maximum and minimum dates
        max_date = max(dates)
        min_date = min(dates)

        # Convert the max_date and min_date back to string format
        max_date_str = max_date.strftime('%m/%d/%Y')
        min_date_str = min_date.strftime('%m/%d/%Y')

        end_date = max_date_str
        start_date = min_date_str
        # convert the list to a dictionary
        ner_dict = {label: value for value, label in ents}
        # print(start_date,end_date,modes)

        revenue = filter_data(start_date, end_date, company_name, saved_msg)

        return revenue


def getResponse(ints, msg, userID='123', pre_msg="hello", show_details=False):
    print("ints>>", ints)

    # If we have a classification, then find the matching intent tag
    if ints:
        # Loop as long as there are matches to process
        while ints:
            for i in intents['intents']:

                # Find a tag matching the first result
                if i['tag'] == ints[0]["intent"]:

                    if 'trigger_lane' in i:
                        response = get_top_bottom_lanes(msg)
                        return response
                    # if 'trigger_cp' in i:
                    #     msg = random.choice(i['responses']).replace("\n", "\n")
                    #     response = company_info(msg)
                    #     return response

                    # Set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details:
                            print('context:', i['context_set'])
                        context[userID] = i['context_set']
                        context[pre_msg] = msg
                        print("context[userID]", context[pre_msg], context[userID])

                    # Check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or (
                            userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details:
                            print('tag:', i['tag'])

                        # Check if trigger condition is met
                        if 'trigger' in i and i['trigger'].lower() == "true":
                            # Join user input and response and call another function
                            # user_input = input("Please provide the period: ")
                            response = random.choice(i['responses']) + msg
                            print("====", response)
                            # Call another function with user_input and response
                            response = generate_result(response)
                            return response
                        if 'trigger_cp' in i and i['trigger_cp'].lower() == "true":
                            print("wala")

                            # Join user input and response and call another function
                            # user_input = input("Please provide the period: ")
                            company_name = context[userID]
                            input_text = msg
                            saved_msg = context[pre_msg]
                            response = generate_result_cpwise(company_name, input_text, saved_msg)
                            print("====", response)
                            # Call another function with user_input and response
                            # response = generate_result(response)
                            return response
                        else:
                            # A random response from the intent
                            print(random.choice(
                                i['responses']).replace("\n", "\n"))

                        return random.choice(i['responses']).replace("\n", "\n")

            ints.pop(0)

