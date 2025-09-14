from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from datasets import Dataset, Features, Value, Image
from wordfreq import top_n_list
import random

NUM_WORDS = 2000
OUTPUT_DIR = "rebus_images_en"
os.makedirs(OUTPUT_DIR, exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("http://rebus1.com/en/index.php?item=rebus_generator")
driver.set_window_size(1920, 1080)

wait = WebDriverWait(driver, 10)

try:
    consent_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "button.fc-cta-consent.fc-primary-button")
        )
    )
    consent_button.click()
    print("✅ Consent button clicked")
except Exception as e:
    print("ℹ️ No consent popup found or already dismissed:", e)

def get_random_word(lang="en", n=5000):
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
        input_box = wait.until(EC.element_to_be_clickable((By.NAME, "slovo")))
        driver.execute_script("arguments[0].scrollIntoView(true);", input_box)
        input_box.clear()
        input_box.send_keys(word)

        button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//input[@type='submit' and @value='Create a Rebus']")
        ))
        driver.execute_script("arguments[0].click();", button)

        rebus_element = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "table[border='0'][height='200'][align='center']")
            )
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", rebus_element)

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
dataset.save_to_disk("rebus_english_rl")
