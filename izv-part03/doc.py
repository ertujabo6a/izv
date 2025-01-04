import pandas as pd
import matplotlib.pyplot as plt

accidents = pd.read_pickle("accidents.pkl.gz")

# **1. Data of weather conditions (p18)**
weather_counts = accidents['p18'].value_counts().sort_index()
weather_labels = {
    1: "Neztížené",
    2: "Mlha",
    3: "Slabý déšť / mrholení",
    4: "Déšť",
    5: "Sněžení",
    6: "Námraza / náledí",
    7: "Nárazový vítr",
    0: "Jiné ztížené"
}
weather_counts.index = weather_counts.index.map(weather_labels)

# **2. Data of visibility conditions (p19)**
visibility_counts = accidents['p19'].value_counts().sort_index()
visibility_labels = {
    1: "Ve dne - nezhoršená",
    2: "Ve dne - zhoršená (svítání, soumrak)",
    3: "Ve dne - zhoršená (mlha, sněžení, déšť)",
    4: "V noci - osvětleno, nezhoršená",
    5: "V noci - osvětleno, zhoršená",
    6: "V noci - neosvětleno, nezhoršená",
    7: "V noci - neosvětleno, zhoršená"
}
visibility_counts.index = visibility_counts.index.map(visibility_labels)

# **3. Calculating values for the text section**
total_accidents = len(accidents)
bad_weather_accidents = accidents[accidents['p18'] != 1].shape[0]
bad_weather_percent = (bad_weather_accidents / total_accidents) * 100
night_accidents = accidents[accidents['p19'] >= 4].shape[0]
day_accidents = accidents[accidents['p19'] < 4].shape[0]
night_percent = (night_accidents / total_accidents) * 100
day_percent = (day_accidents / total_accidents) * 100
worsened_visibility_day = visibility_counts.loc["Ve dne - zhoršená (svítání, soumrak)"] + \
                          visibility_counts.loc["Ve dne - zhoršená (mlha, sněžení, déšť)"]
worsened_visibility_day_percent = (worsened_visibility_day / total_accidents) * 100

# **4. Generate bar chart for weather conditions**
plt.figure(figsize=(10, 6))
weather_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Počet nehod za různých povětrnostních podmínek")
plt.ylabel("Počet nehod")
plt.xlabel("Povětrnostní podmínky")
plt.tight_layout()
plt.savefig("fig.png")
plt.show()

# **5. Print tabular data in plain text format**
print("\nTabulka: Počet nehod podle viditelnosti")
print("=======================================")
print(f"{'Viditelnost':40} {'Počet nehod':>10}")
print("-" * 51)
for index, count in visibility_counts.items():
    print(f"{index:40} {count:>10}")
print("-" * 51)

# **6. Print key statistics to the standard output**
print("\n=== Statistická analýza ===")
print(f"Celkový počet nehod: {total_accidents}")
print(f"Nehod za nepříznivého počasí: {bad_weather_accidents} ({bad_weather_percent:.2f}%)")
print(f"Nehod ve dne: {day_accidents} ({day_percent:.2f}%)")
print(f"Nehod v noci: {night_accidents} ({night_percent:.2f}%)")
print(f"Nehod za zhoršené viditelnosti ve dne (svítání, soumrak, mlha, déšť): {worsened_visibility_day} "
      f"({worsened_visibility_day_percent:.2f}%)")
