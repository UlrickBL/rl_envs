from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

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

input_box = driver.find_element(By.CLASS_NAME, "champ_rebus")
input_box.send_keys("éléphant")

button = driver.find_element(By.ID, "rollover_abracadabra")
button.click()

time.sleep(2)

rebus_element = driver.find_element(By.ID, "rebus")
rebus_element.screenshot("rebus.png")

driver.quit()
