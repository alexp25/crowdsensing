import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
price_df = pd.read_csv("polymarket_price_daily.csv", parse_dates=['date'])
events_df = pd.read_csv("events.csv", parse_dates=['date'])
poll_df = pd.read_csv("polling_data.csv", parse_dates=['date'])

# Calculate daily price changes
price_df['Simion Change'] = price_df['George Simion'].diff()
price_df['Dan Change'] = price_df['Nicușor Dan'].diff()

# Filter for May 2025
price_may = price_df[(price_df['date'] >= '2025-04-15') & (price_df['date'] <= '2025-05-31')]
events_may = events_df[(events_df['date'] >= '2025-04-15') & (events_df['date'] <= '2025-05-31')]


# Merge polling with price data (optional for deeper analysis)
merged_df = pd.merge(price_df, poll_df, on='date', how='left')

# Plot static overlay chart with numbered events
plt.figure(figsize=(14, 6))
# plt.plot(price_df['date'], price_df['George Simion'], label='George Simion', linewidth=4)
# plt.plot(price_df['date'], price_df['Nicușor Dan'], label='Nicușor Dan', linewidth=4)

plt.plot(price_may['date'], price_may['Nicușor Dan'], label='Nicușor Dan', color='blue', linewidth=2)
plt.plot(price_may['date'], price_may['George Simion'], label='George Simion', color='orange', linewidth=2)


# Number the events and annotate with numbers only
event_labels = []
for idx, row in events_df.iterrows():
    color = {
        'disinfo': 'red',
        'rational': 'green',
        'campaign': 'blue'
    }.get(row['type'], 'gray')
    
    plt.axvline(x=row['date'], color=color, linestyle='--', alpha=0.6)
    plt.text(row['date'], 0.55, str(row["event_number"]), rotation=90, fontsize=12, color=color, verticalalignment='center')

# plt.xticks(price_may['date'], price_may['date'].dt.strftime('%Y-%m-%d'), rotation=90)
plt.xticks(price_may['date'], price_may['date'].dt.strftime('%m-%d'), rotation=90)

plt.title("Polymarket Price Trends with Numbered Events", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Polymarket Price", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig("plot.png", dpi=300)

# Print numbered event legend
print("\nEvent Legend:")
for label in event_labels:
    print(label)
