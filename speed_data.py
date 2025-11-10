# ...existing code...
from encoded_motor_driver import EncodedMotorDriver
from utime import sleep
import utime
from machine import Pin

motor = EncodedMotorDriver(driver_ids=(10, 9, 11), #pwm_id, in1_id, in2_id
        encoder_ids=(17, 16),)
stby_pin = Pin(12, Pin.OUT)
stby_pin.on()

# Open CSV and write header (overwrites existing file each run)
with open("speed_data.csv", "w") as f:
    f.write("timestamp_ms,baud_rate,measured_velocity\n")

    try:
        for i in range(0, 65535, 100):
            motor.forward(i)
            sleep(0.1)  # allow motor/encoder to settle
            vel = motor.meas_lin_vel
            line = "{},{}\n".format(i, vel)
            f.write(line)
            f.flush()  # ensure data is written to storage

    finally:
        motor.forward(0)
# ...existing code...