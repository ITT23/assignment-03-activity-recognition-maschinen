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




@window.event
def on_draw():
    window.clear()
    if sensor.get_capabilities():
        if sensor.has_capability('accelerometer'):
            accelerometer_data = sensor.get_value('accelerometer')
        else:
            return
        if sensor.has_capability('gyroscope'):
            gyroscope_data = sensor.get_value('gyroscope')
        else:
            return
        if sensor.has_capability('gravity'):
            gravity_data = sensor.get_value('gravity')
        else:
            return
        visualizer.update(accelerometer_data, gyroscope_data, gravity_data)
        visualizer.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:
        sensor.disconnect()
        sys.exit(0)


if __name__ == '__main__':
    print("start")
    trainer = Trainer()
    trainer.train()
    visualizer = Visualizer(trainer.classifier)
    pyglet.app.run()
