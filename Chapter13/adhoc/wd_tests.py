#!/usr/bin/env python3
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=640,480")


driver = webdriver.Chrome(chrome_options=options)
t = time.time()
#driver.get("http://localhost:5000")
driver.get("http://www.yandex.ru")
#driver.get("http://www3.hilton.com/en/index.html?ignoreGateway=true")
t_get = time.time() - t
print("Page opened in %s seconds" % t_get)
driver.get_screenshot_as_file("p.png")
print("Screenshot taken!")
driver.close()
