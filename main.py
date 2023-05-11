import pyglet
import sys
import config
from activity_visualizer import Visualizer
from DIPPID import SensorUDP
from training import Trainer

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 900

sensor = SensorUDP(config.PORT)
window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
pyglet.gl.glClearColor(0.1, 0.6, 0.2, 0)

trainer = Trainer()
trainer.train()
visualizer = Visualizer(trainer.classifier)


@window.event
def on_draw():
    window.clear()
    if sensor.get_capabilities():
        accelerometer_data = sensor.get_value('accelerometer')
        gyroscope_data = sensor.get_value('gyroscope')
        gravity_data = sensor.get_value('gravity')
        visualizer.update(accelerometer_data, gyroscope_data, gravity_data)
    visualizer.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        sensor.disconnect()
        sys.exit(0)


if __name__ == '__main__':
    pyglet.app.run()
