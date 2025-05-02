import webbrowser
import time
import pyautogui

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
webbrowser.open(url)

time.sleep(3)

pyautogui.click(x=500, y=500)