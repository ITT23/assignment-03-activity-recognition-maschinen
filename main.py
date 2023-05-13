import pyglet
import sys
import config
from activity_visualizer import Visualizer
from DIPPID import SensorUDP
from training import Trainer

trainer = Trainer()
trainer.train()

sensor = SensorUDP(config.PORT)
window = pyglet.window.Window(width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT)
pyglet.gl.glClearColor(config.BACKGROUND_COLOR_R, config.BACKGROUND_COLOR_G, config.BACKGROUND_COLOR_B, config.BACKGROUND_COLOR_T)
# fps_display = pyglet.window.FPSDisplay(window=window)


@window.event
def on_draw():
    if window:
        window.clear()
        # fps_display.draw()
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

    visualizer = Visualizer(trainer.classifier)
    pyglet.app.run()
