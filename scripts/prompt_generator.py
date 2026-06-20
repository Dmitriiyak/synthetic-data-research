import itertools
import random
import pandas as pd

random.seed(42)

LANDMARKS = {
    "burj_khalifa": "Burj Khalifa",
    "chichen_itza": "Chichen Itza",
    "christ_the_reedemer": "Christ the Redeemer",
    "eiffel_tower": "Eiffel Tower",
    "great_wall_of_china": "Great Wall of China",
    "pyramids_of_giza": "Pyramids of Giza",
    "roman_colosseum": "Roman Colosseum",
    "statue_of_liberty": "Statue of Liberty"
}

ANGLES = [
    "front view",
    "side view",
    "three-quarter view",
    "street level view",
    "aerial view"
]

DISTANCES = [
    "close-up",
    "medium distance",
    "long distance"
]

LIGHTING = [
    "sunrise",
    "daylight",
    "sunset"
]

WEATHER = [
    "clear sky",
    "cloudy",
    "overcast",
    "rainy weather"
]

BASE_PROMPT = (
    "{landmark}, "
    "{angle}, "
    "{distance}, "
    "{lighting}, "
    "{weather}, "
    "professional travel photography, "
    "realistic photo, "
    "DSLR camera, "
    "highly detailed, "
    "natural colors"
)

all_combinations = list(
    itertools.product(
        ANGLES,
        DISTANCES,
        LIGHTING,
        WEATHER
    )
)

rows = []

prompt_id = 1

for class_name, landmark in LANDMARKS.items():

    selected = random.sample(all_combinations, 12)

    for split, combinations in {
        "train": selected[:10],
        "test": selected[10:]
    }.items():

        for angle, distance, lighting, weather in combinations:

            prompt = BASE_PROMPT.format(
                landmark=landmark,
                angle=angle,
                distance=distance,
                lighting=lighting,
                weather=weather
            )

            rows.append({
                "prompt_id": prompt_id,
                "class_name": class_name,
                "split": split,
                "angle": angle,
                "distance": distance,
                "lighting": lighting,
                "weather": weather,
                "prompt": prompt
            })

            prompt_id += 1

df = pd.DataFrame(rows)

df.to_csv(
    "prompt_templates.csv",
    index=False
)

print(df.head())
print(f"Total prompts: {len(df)}")