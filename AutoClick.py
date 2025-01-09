import time
import pyautogui

def click_after_seconds(x, y, seconds):
    """Clicks at the specified coordinates after a given number of seconds."""
    pyautogui.click(x, y)
    time.sleep(seconds)


if __name__ == "__main__":
    x_coordinate = -2097  # Replace with your desired x-coordinate
    y_coordinate = 887  # Replace with your desired y-coordinate
    delay_seconds = 11   # Replace with your desired delay in seconds
    while 50:
        click_after_seconds(x_coordinate, y_coordinate, delay_seconds)
