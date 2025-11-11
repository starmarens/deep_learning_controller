from machine import Pin, PWM


class MotorDriver:
    def __init__(self, pwm_id, in1_id, in2_id) -> None:
        self.pwm_pin = PWM(Pin(pwm_id))
        self.pwm_pin.freq(1000)
        self.in1_pin = Pin(in1_id, Pin.OUT)
        self.in2_pin = Pin(in2_id, Pin.OUT)

    def stop(self):
        self.pwm_pin.duty_u16(0)

    def forward(self, speed):  # map 0~65535 to 0~1
        self.in1_pin.off()
        self.in2_pin.on()
        self.pwm_pin.duty_u16(speed)

    def backward(self, speed):  # map 0~65535 to 0~1
        self.in1_pin.on()
        self.in2_pin.off()
        self.pwm_pin.duty_u16(speed)


# TEST
if __name__ == "__main__":
    from time import sleep

    # SETUP
    # md = MotorDriver(pwm_id=9, in1_id=11, in2_id=10)  # right motor
    md = MotorDriver(pwm_id=15, in1_id=13, in2_id=14)  # left motor
    STBY = Pin(12, Pin.OUT)
    STBY.off()

    # LOOP
    STBY.on()  # enable motor driver
    # Forwardly ramp up and down
    for i in range(60000):
        md.forward(i)
        print(f"f, dc: {i}%")
        sleep()  # 4 seconds to ramp up
    for i in reversed(range(100)):
        md.forward((i + 1) / 100)
        print(f"f, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp down
    # Backwardly ramp up and down
    for i in range(100):
        md.backward((i + 1) / 100)
        print(f"b, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp up
    for i in reversed(range(100)):
        md.backward((i + 1) / 100)
        print(f"b, dc: {i}%")
        sleep(4 / 100)  # 4 seconds to ramp down

    # Terminate
    md.stop()
    print("motor stopped.")
    sleep(0.1)  # full stop
    STBY.off()  # disable motor driver
    print("motor driver disabled.")
