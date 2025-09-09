from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from datasets import Dataset, Features, Value, Image
from wordfreq import top_n_list
import random

NUM_WORDS = 2000
OUTPUT_DIR = "rebus_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--remote-debugging-port=9222")

options.binary_location = "/usr/bin/google-chrome"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("http://www.rebus-o-matic.com/")
driver.set_window_size(1920, 1080)

def get_random_word(lang="fr", n=5000):
    words = top_n_list(lang, n)
    return random.choice(words)

data = []
seen_words = set()

for i in range(NUM_WORDS):
    word = get_random_word()
    if word in seen_words:
        continue
    seen_words.add(word)

    try:
        input_box = driver.find_element(By.CLASS_NAME, "champ_rebus")
        input_box.clear()
        input_box.send_keys(word)

        button = driver.find_element(By.ID, "rollover_abracadabra")
        button.click()

        time.sleep(2)  # wait for generation

        rebus_element = driver.find_element(By.ID, "rebus")
        img_path = os.path.join(OUTPUT_DIR, f"{word}.png")
        rebus_element.screenshot(img_path)

        data.append({"word": word, "image": img_path})
        print(f"[{i+1}/{NUM_WORDS}] {word} ✅")

    except Exception as e:
        print(f"Error with word {word}: {e}")
        continue

driver.quit()

features = Features({
    "word": Value("string"),
    "image": Image()
})

dataset = Dataset.from_list(data, features=features)

dataset.save_to_disk("rebus_french_rl")
